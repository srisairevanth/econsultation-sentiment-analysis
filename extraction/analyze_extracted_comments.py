import os
import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze Regulations.gov comments text quality.")
    parser.add_argument("file_path", nargs="?", default="raw_financial_comments_large.json", help="Path to comments JSON file (default: raw_financial_comments_large.json)")
    args = parser.parse_args()
    
    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        sys.exit(1)
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            comments = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)
        
    total_comments = len(comments)
    if total_comments == 0:
        print("No comments found in the file.")
        sys.exit(0)
        
    substantive_comments = []
    attachment_comments = []
    null_or_empty_comments = []
    
    attachment_keywords = ["attached", "attachment", "pdf", "letter", "docx", "submit", "behalf of"]
    
    for c in comments:
        cid = c.get("id")
        attr = c.get("attributes", {})
        comment_text = attr.get("comment")
        
        if not comment_text:
            null_or_empty_comments.append(cid)
        else:
            cleaned = comment_text.strip().lower()
            if any(kw in cleaned for kw in attachment_keywords):
                attachment_comments.append((cid, comment_text))
            else:
                substantive_comments.append((cid, comment_text))
                
    substantive_count = len(substantive_comments)
    attachment_count = len(attachment_comments)
    null_count = len(null_or_empty_comments)
    
    print("=" * 60)
    print("           Regulations.gov Comment Quality Analysis Report")
    print("=" * 60)
    print(f"Total Comments Processed: {total_comments}\n")
    print(f"1. Substantive Direct Text Comments: {substantive_count} ({substantive_count / total_comments * 100:.1f}%)")
    print("   (Contains actual opinion/text directly typed in, ready for sentiment analysis)")
    print(f"2. Attachment Placeholders:         {attachment_count} ({attachment_count / total_comments * 100:.1f}%)")
    print("   (Refers to an uploaded file or document)")
    print(f"3. Null / Empty Comments:           {null_count} ({null_count / total_comments * 100:.1f}%)\n")
    
    print("-" * 60)
    print("Substantive Comments - Text Samples:")
    print("-" * 60)
    
    # Print up to 5 samples
    for i, (cid, text) in enumerate(substantive_comments[:5], start=1):
        clean_text = text.replace("<br/>", "\n").replace("&rsquo;", "'").replace("&mdash;", "—").strip()
        # Truncate for display
        truncated_text = clean_text[:200] + "..." if len(clean_text) > 200 else clean_text
        print(f"Sample {i} [{cid}]:\n{truncated_text}\n")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
