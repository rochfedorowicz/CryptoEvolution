import os
import glob

html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Documentation</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f9;
        }}
        .container {{
            width: 80%;
            margin: auto;
            overflow: hidden;
        }}
        header {{
            background: #333;
            color: #fff;
            padding-top: 30px;
            min-height: 70px;
            border-bottom: #77aaff 3px solid;
        }}
        header h1 {{
            text-align: center;
            text-transform: uppercase;
            margin: 0;
        }}
        nav {{
            margin-top: 10px;
        }}
        nav a {{
            color: #333;
            text-decoration: none;
            text-transform: uppercase;
            margin: 0 10px;
        }}
        .content {{
            padding: 20px;
            background: #fff;
            margin-top: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .content h2 {{
            text-align: center;
            color: #333;
        }}
        .content p {{
            text-align: center;
            font-size: 1.2em;
        }}
        .content a {{
            color: #007bff;
            text-decoration: none;
        }}
        .content a:hover {{
            text-decoration: underline;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            background: #333;
            color: #fff;
            margin-top: 20px;
            border-top: #77aaff 3px solid;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Welcome to the Project Documentation</h1>
    </header>
    <div class="container">
        <div class="content">
            <h2>Latest Documentation</h2>
            <p><a href="docs/html/index.html">Latest Documentation</a></p>
            <h2>Previous Reports</h2>
            {links}
        </div>
    </div>
    <footer>
        <p>&copy; 2024 Project Documentation</p>
    </footer>
</body>
</html>
'''

if __name__ == "__main__":
    report_files = glob.glob('gh-pages/reports/**/report.html', recursive=True)
    links = ''.join(f'<p><a href="{rel_path}">{rel_path}</a></p>' for report in report_files if (rel_path := os.path.relpath(report, "gh-pages")))
    html_content = html_template.format(links=links)
    print(html_content)
