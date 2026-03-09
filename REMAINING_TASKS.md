# TrollGuard – Remaining Tasks for Submission

Before final submission, complete the following:

## 1. Fill Placeholders in Report

Edit `TrollGuard_Project_Report.md` and replace:

- `[Your Full Name]` – Your name
- `[Your Roll No.]` – Roll / Register number
- `[Guide Name]` – Guide’s name
- `[Guide Designation]` – e.g. Assistant Professor, Department of Computer Science
- `[e.g. 2024 – 2025]` – Academic year
- `[College Name]` – Full college name
- `[University Name]` – University (if applicable for affiliation)
- `[e.g. Streamlit Cloud / Railway / None]` – Deployment target (if you use one)

## 2. Table of Contents

Add a Table of Contents (TOC) to the report. In Word: References → Table of Contents → Automatic.

For Markdown, you can add:

```markdown
## Table of Contents
1. [Introduction](#chapter-1--introduction)
2. [System Study](#chapter-2--system-study)
...
```

## 3. Screenshots

Insert screenshots in the report and presentation:

- Class distribution plot
- Confusion matrix
- Streamlit app (Predict, Chat analysis, Train)
- Sample output (predictions CSV, chat summary)

## 4. Signatures

Get physical signatures for:

- Bonafide Certificate (HoD)
- Declaration (Student)

## 5. Presentation

- Export `TrollGuard_Presentation.md` to PPTX using Marp CLI:  
  `marp TrollGuard_Presentation.md -o TrollGuard_Presentation.pptx`
- Or use the Marp VS Code extension and export to PowerPoint
- Replace `[College Name]` and `[Academic Year]` in the slides

## 6. Deployment (Optional)

If deploying to Streamlit Cloud:

- Ensure `requirements.txt` is correct
- Either commit `outputs/tfidf.joblib` and `outputs/logreg_model.joblib`, or  
  train the model from the app’s **Train model** tab after deploy
- Test the live app URL before submission

## 7. DOCX Export

If you need to regenerate the DOCX:

```bash
pandoc TrollGuard_Project_Report.md -o TrollGuard_Project_Report.docx
```

## Checklist

- [ ] Placeholders filled
- [ ] TOC added
- [ ] Screenshots added
- [ ] Signatures obtained
- [ ] Presentation exported to PPTX
- [ ] Deployment tested (if applicable)
