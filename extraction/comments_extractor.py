import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Base configuration
BASE_URL = "https://api.regulations.gov/v4/comments"

def parse_args():
    parser = argparse.ArgumentParser(description="Extract raw public comments from Regulations.gov API v4.")
    parser.add_argument("--docket-id", dest="docket_id", help="Filter by Docket ID (e.g., EOIR-2020-0003)")
    parser.add_argument("--agency-id", dest="agency_id", help="Filter by Agency Acronym (e.g., EPA, FDA)")
    parser.add_argument("--search-term", dest="search_term", help="Search term filter")
    parser.add_argument("--posted-date-ge", dest="posted_date_ge", help="Filter by posted date >= (YYYY-MM-DD)")
    parser.add_argument("--posted-date-le", dest="posted_date_le", help="Filter by posted date <= (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Maximum number of comments to extract")
    parser.add_argument("--output", default="raw_comments.json", help="Path to save the JSON output file (default: raw_comments.json)")
    parser.add_argument("--metadata-only", dest="metadata_only", action="store_true", help="Only fetch comment headers/metadata list (faster, but excludes full comment body)")
    return parser.parse_args()

def iso_to_api_date(iso_str, subtract_hours=6):
    """
    Converts ISO 8601 UTC timestamp from the API to the Regulations.gov filter format (yyyy-MM-dd HH:mm:ss).
    Subtracts a buffer (default 6 hours) to account for the API's Eastern Time timezone requirement
    for filters and ensure no comments are missed at pagination page-20 boundaries.
    """
    if not iso_str:
        return None
    if iso_str.endswith('Z'):
        iso_str = iso_str[:-1]
    
    # Strip microsecond parts if present
    dt_part = iso_str.split('.')[0]
    
    try:
        dt = datetime.strptime(dt_part, "%Y-%m-%dT%H:%M:%S")
        if subtract_hours:
            dt = dt - timedelta(hours=subtract_hours)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error parsing date {iso_str}: {e}", file=sys.stderr)
        return None

def fetch_comment_detail(api_key, comment_id):
    """
    Fetches detailed representation of a specific comment to retrieve the actual comment text.
    Retries indefinitely with backoff on 429 rate limit errors to ensure no text body is lost.
    """
    url = f"{BASE_URL}/{comment_id}"
    headers = {"X-Api-Key": api_key}
    backoff = 30  # Start sleeping at 30 seconds for rate limits
    
    while True:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                res_json = response.json()
                return res_json.get("data")
            elif response.status_code == 429:
                # Attempt to read Retry-After header
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_time = int(retry_after) + 2
                    except ValueError:
                        sleep_time = 300
                else:
                    sleep_time = backoff
                    backoff = min(backoff * 2, 600)  # Sleep up to 10 minutes max
                
                print(f"[429] Rate limit hit on detail fetch for {comment_id}. Sleeping for {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"Error fetching details for {comment_id}: API status {response.status_code}", file=sys.stderr)
                return None
        except requests.RequestException as e:
            print(f"Connection error fetching details for {comment_id}: {e}. Retrying in 15s...", file=sys.stderr)
            time.sleep(15)


def fetch_comments(api_key, filters, limit=None, metadata_only=False):
    headers = {
        "X-Api-Key": api_key
    }
    
    list_comments = []
    seen_ids = set()
    
    # Base query parameters (JSON:API standard)
    params = {
        "page[size]": 250,
        "sort": "lastModifiedDate"
    }
    
    # Add optional search filters
    if filters.get("docket_id"):
        params["filter[docketId]"] = filters["docket_id"]
    if filters.get("agency_id"):
        params["filter[agencyId]"] = filters["agency_id"]
    if filters.get("search_term"):
        params["filter[searchTerm]"] = filters["search_term"]
    if filters.get("posted_date_ge"):
        params["filter[postedDate][ge]"] = filters["posted_date_ge"]
    if filters.get("posted_date_le"):
        params["filter[postedDate][le]"] = filters["posted_date_le"]
        
    last_modified_ge = None
    cycle_count = 0
    reached_limit = False
    
    # --- PHASE 1: Paginate and collect list metadata ---
    print("\n--- PHASE 1: Fetching comments list/metadata ---")
    while not reached_limit:
        cycle_count += 1
        if last_modified_ge:
            params["filter[lastModifiedDate][ge]"] = last_modified_ge
            
        print(f"\n--- Starting List Fetch Cycle #{cycle_count} ---")
        print(f"Active Query Parameters: {params}")
        
        cycle_has_new_data = False
        
        # Paginate page 1 to 20 (API constraint of max 20 pages per query)
        for page_num in range(1, 21):
            params["page[number]"] = page_num
            
            # Rate limit/Transient error retry block
            backoff = 30  # Start backoff at 30 seconds for rate limits in Phase 1
            network_retries = 0
            max_network_retries = 5
            response = None
            
            while True:
                try:
                    response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
                    
                    # Rate limit headers check
                    limit_remaining = response.headers.get("X-RateLimit-Remaining")
                    limit_total = response.headers.get("X-RateLimit-Limit")
                    if limit_remaining is not None:
                        print(f"Rate Limit: {limit_remaining}/{limit_total} requests remaining.")
                    
                    if response.status_code == 200:
                        break
                    elif response.status_code == 429:
                        # Attempt to read Retry-After header
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                sleep_time = int(retry_after) + 2
                            except ValueError:
                                sleep_time = 300
                        else:
                            sleep_time = backoff
                            backoff = min(backoff * 2, 600)  # Sleep up to 10 minutes max
                            
                        print(f"[429] Rate limit hit. Sleeping for {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        print(f"API Error {response.status_code}: {response.text}", file=sys.stderr)
                        response.raise_for_status()
                except requests.RequestException as e:
                    network_retries += 1
                    print(f"Request connection error on page {page_num} (Retry {network_retries}/{max_network_retries}): {e}", file=sys.stderr)
                    if network_retries >= max_network_retries:
                        raise
                    time.sleep(15)

            
            if not response or response.status_code != 200:
                print("Failed to fetch data from API.", file=sys.stderr)
                break
                
            res_json = response.json()
            data = res_json.get("data", [])
            
            if not data:
                print("No comments returned in page response.")
                break
                
            new_comments_in_page = 0
            for comment in data:
                comment_id = comment.get("id")
                if comment_id and comment_id not in seen_ids:
                    seen_ids.add(comment_id)
                    list_comments.append(comment)
                    new_comments_in_page += 1
                    cycle_has_new_data = True
                    
                    if limit and len(list_comments) >= limit:
                        print(f"\nTarget limit of {limit} comments reached in list phase.")
                        reached_limit = True
                        break
            
            if reached_limit:
                break
                
            print(f"Page {page_num}: Retrieved {len(data)} comment metadata headers ({new_comments_in_page} new). Total collected: {len(list_comments)}")
            
            # If response returned fewer records than standard page size, we've matched everything.
            if len(data) < 250:
                print("Retrieved all matched comment metadata headers.")
                reached_limit = True
                break
                
            # If we reached page 20, we must slide the date window to get more results
            if page_num == 20:
                last_record = data[-1]
                last_modified_val = last_record.get("attributes", {}).get("lastModifiedDate")
                if last_modified_val:
                    formatted_date = iso_to_api_date(last_modified_val)
                    if formatted_date:
                        last_modified_ge = formatted_date
                        print(f"\n[Page 20 Cap Reached] Sliding window. New lastModifiedDate >= {last_modified_ge}")
                    else:
                        print("Could not format date for next cycle. Stopping.", file=sys.stderr)
                        reached_limit = True
                        break
                else:
                    print("lastModifiedDate missing in page 20 last record. Stopping.", file=sys.stderr)
                    reached_limit = True
                    break
                    
        # Safeguard to advance window if cycle found no new items
        if not reached_limit and not cycle_has_new_data and last_modified_ge:
            try:
                dt = datetime.strptime(last_modified_ge, "%Y-%m-%d %H:%M:%S")
                dt = dt + timedelta(seconds=1)
                last_modified_ge = dt.strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[Safeguard] Advancing sliding window by 1 second to: {last_modified_ge}")
            except Exception as e:
                print(f"Error advancing time window: {e}", file=sys.stderr)
                break

    # If only metadata is requested, return now
    if metadata_only:
        return list_comments

    # --- PHASE 2: Fetch details (full body content) for each comment ---
    print(f"\n--- PHASE 2: Fetching full details for {len(list_comments)} comments ---")
    detailed_comments = []
    
    for idx, comment in enumerate(list_comments, start=1):
        comment_id = comment.get("id")
        print(f"[{idx}/{len(list_comments)}] Fetching full details for comment ID: {comment_id}...")
        
        detail = fetch_comment_detail(api_key, comment_id)
        if detail:
            detailed_comments.append(detail)
        else:
            # Fall back to list metadata if detail fetch fails
            print(f"Warning: Failed to fetch detail for {comment_id}. Falling back to list metadata.", file=sys.stderr)
            detailed_comments.append(comment)
            
    # Report stats on how many have body text
    comments_with_text = sum(1 for c in detailed_comments if c.get("attributes", {}).get("comment") is not None)
    print(f"\nDetails retrieval complete. Total: {len(detailed_comments)}. Comments containing text body: {comments_with_text}")
    
    return detailed_comments

def main():
    # Load environment variables from the project root's .env, regardless of CWD
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    api_key = os.getenv("REGULATIONS_API_KEY")
    if not api_key:
        print("Error: REGULATIONS_API_KEY environment variable not found. Please set it in a .env file or your shell environment.", file=sys.stderr)
        sys.exit(1)
        
    args = parse_args()
    
    # Validation warning
    if not (args.docket_id or args.agency_id or args.search_term or args.posted_date_ge or args.posted_date_le):
        print("WARNING: No search filters (docket, agency, search term, or dates) were specified.")
        print("Extracting all comments from Regulations.gov will consume significant API quota.")
        print("Press Ctrl+C to cancel or let the script run...")
        
    filters = {
        "docket_id": args.docket_id,
        "agency_id": args.agency_id,
        "search_term": args.search_term,
        "posted_date_ge": args.posted_date_ge,
        "posted_date_le": args.posted_date_le
    }
    
    print("Initializing Regulations.gov Comment Extractor...")
    print(f"Destination file: {args.output}")
    if args.limit:
        print(f"Extraction Limit: {args.limit} comments")
    if args.metadata_only:
        print("Mode: Metadata-Only (speeds up run, but comment body text will be omitted)")
    else:
        print("Mode: Full Details (fetches metadata first, then details to retrieve comment text)")
        
    start_time = time.time()
    try:
        comments = fetch_comments(api_key, filters, limit=args.limit, metadata_only=args.metadata_only)
        
        # Save fetched comments exactly as received from the API (nested list structure)
        print(f"\nSaving {len(comments)} raw comments to {args.output}...")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comments, f, indent=2, ensure_ascii=False)
            
        duration = time.time() - start_time
        print(f"Extraction complete! Saved to {args.output} in {duration:.2f} seconds.")
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user. Exiting.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nExtraction failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
