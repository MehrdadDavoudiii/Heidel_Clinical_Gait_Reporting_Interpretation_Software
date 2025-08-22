# Motion Capture Gait Analysis Report Generator


**Author:** Mehrdad Davoudi (PhD student, Heidelberg University, Germany)

This project reads an anonymized gait C3D file (Heidel_file.c3d), extracts the
gait data, plots kinematics/kinetics into a Word report, and optionally
uploads the (anonymized) content to GPT-5 via API to append a clinical
description and produce the final PDF. The goal is to provide a concise
summary that reduces subjectivity in clinical decision-making.

DATA NOTE
---------
• The kinematics/kinetics patterns in Heidel_file.c3d are from a healthy subject.
• The spatiotemporal parameters in ST.txt (cadence, speed, foot-off, single/double support)
  are from a different patient dataset (not embedded in the C3D). This intentional mismatch
  is used to test and demonstrate the interpretation step.

Pipeline
--------
1) Convert_c3d_to_json.py
   - Reads the .c3d and exports angles.json, moments.json, powers.json.

2) Make_report.py
   - Reads the JSON files + ST.txt (cadence, speed, foot-off, single/double support).
   - Builds motion_report.docx with:
       • Cover page placeholders
       • Weight/Height
       • Spatiotemporal table
       • Kinematics page
       • Kinetics page

3) Auto_interpret_report.py
   - Sends the report text to the ChatGPT API for a professional, concise
     interpretation and clinical conclusion (commands in code).
   - Saves: motion_report_with_conclusion.pdf

Run All Steps
-------------
    python run_pipeline.py
    (you will be asked for your OPENAI_API_KEY)
    

Requirements
------------
Install dependencies first:
    pip install -r requirements.txt

Getting an OpenAI API Key (for the optional interpretation step)
----------------------------------------------------------------
1) Visit:
   https://platform.openai.com
2) Sign in (or create an account), then add **Pay-As-You-Go billing**.
   (Note: ChatGPT Plus is separate and does not include API credits.)
3) Go to **API keys** → **Create new secret key**. You’ll get a key like `sk-...`.
4) Keep it private. Do not commit it to Git. Provide it to the pipeline when prompted
   or set it as an environment variable (e.g., `OPENAI_API_KEY`).

Notes
-----
• The included C3D is anonymized and consented for research.
• docx2pdf requires Microsoft Word on Windows
• API usage costs are billed by OpenAI; typical runs are a few cents.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
