#!/usr/bin/env python3
import os
import subprocess
import re
from pathlib import Path
from html.parser import HTMLParser

class DocxToMarkdownConverter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.markdown = []
        self.current_text = ""
        self.in_bold = False
        self.in_paragraph = False
        self.is_title = False
        self.is_heading = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "p":
            self.in_paragraph = True
            # Check the class to determine formatting
            if 'class' in attrs_dict:
                class_name = attrs_dict['class']
                if 'p1' in class_name:  # Title
                    self.is_title = True
                elif 'p3' in class_name:  # Heading
                    self.is_heading = True
        elif tag == "b":
            self.in_bold = True

    def handle_endtag(self, tag):
        if tag == "p":
            if self.is_title:
                # Main title - use #
                text = self.current_text.strip()
                if text:
                    self.markdown.append(f"# {text}\n")
                self.is_title = False
            elif self.is_heading:
                # Section heading - use ##
                text = self.current_text.strip()
                if text:
                    self.markdown.append(f"\n## {text}\n")
                self.is_heading = False
            else:
                # Regular paragraph
                text = self.current_text.strip()
                if text:
                    self.markdown.append(f"\n{text}\n")
            self.current_text = ""
            self.in_paragraph = False
        elif tag == "b":
            self.in_bold = False

    def handle_data(self, data):
        if self.in_paragraph:
            # Clean up the text
            text = data.strip()
            if text:
                if self.in_bold and not (self.is_title or self.is_heading):
                    self.current_text += f"**{text}**"
                else:
                    self.current_text += text + " "

    def get_markdown(self):
        return "\n".join(self.markdown).strip()

def convert_docx_to_markdown(docx_path, output_path):
    """Convert a docx file to markdown format"""
    # Create temp HTML file
    temp_html = "/tmp/temp_convert.html"

    # Convert docx to HTML using textutil
    subprocess.run([
        'textutil', '-convert', 'html',
        docx_path, '-output', temp_html
    ], capture_output=True, text=True)

    # Read HTML and convert to markdown
    with open(temp_html, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Parse HTML to markdown
    parser = DocxToMarkdownConverter()
    parser.feed(html_content)
    markdown_content = parser.get_markdown()

    # Write markdown file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    # Clean up temp file
    os.remove(temp_html)

    return output_path

def main():
    # Source directory with docx files
    source_dir = Path("/Users/oliverstern/Downloads/Blogs-Articles")
    # Target directory for markdown files
    target_dir = Path("/Users/oliverstern/lemkin_website/articles")
    target_dir.mkdir(exist_ok=True)

    # Find all docx files
    docx_files = list(source_dir.glob("*.docx"))

    print(f"Found {len(docx_files)} .docx files to convert")

    converted_files = []
    for docx_file in docx_files:
        # Create markdown filename (remove special chars from filename)
        base_name = docx_file.stem
        # Clean filename for use as markdown
        clean_name = re.sub(r'[^\w\s-]', '', base_name)
        clean_name = re.sub(r'[-\s]+', '-', clean_name)
        md_filename = f"{clean_name}.md"
        output_path = target_dir / md_filename

        print(f"Converting: {docx_file.name}")
        print(f"  -> {md_filename}")

        try:
            convert_docx_to_markdown(str(docx_file), str(output_path))
            converted_files.append({
                'original': docx_file.name,
                'markdown': md_filename,
                'path': str(output_path)
            })
            print(f"  ✓ Converted successfully")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n Successfully converted {len(converted_files)} files")

    # Save conversion mapping for reference
    import json
    mapping_file = target_dir / "conversion_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(converted_files, f, indent=2)

    print(f"Conversion mapping saved to: {mapping_file}")

if __name__ == "__main__":
    main()