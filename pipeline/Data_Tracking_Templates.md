# DATA TRACKING SPREADSHEET TEMPLATE

## Sheet 1: Image Database

| Image_ID | Source | Category | Type | Collection_Date | Format | File_Size_KB | Original_Dimensions | Verified_Authentic | Generator_Model | Comments |
|----------|--------|----------|------|-----------------|--------|--------------|---------------------|-------------------|-----------------|----------|
| IMG_001 | Unsplash | Portrait | Real | 2024-01-15 | JPG | 245 | 2400x1600 | Yes | N/A | Professional photo |
| IMG_002 | DALL-E | Portrait | AI | 2024-01-15 | PNG | 512 | 1024x1024 | Yes | DALL-E 3 | Prompt-based |
| IMG_003 | Reuters | Landscape | Real | 2024-01-16 | JPG | 890 | 4000x2667 | Yes | N/A | News photo |
| IMG_004 | Midjourney | Landscape | AI | 2024-01-16 | PNG | 478 | 1344x896 | Yes | Midjourney v6 | Artistic style |

---

## Sheet 2: Visual Analysis Results

| Image_ID | Analyst | Analysis_Date | Hands_Assessment | Text_Clarity | Eye_Consistency | Physics_Valid | Background_Quality | Overall_Artifact_Score | Manual_Classification | Confidence_1_10 | Notes |
|----------|---------|----------------|------------------|--------------|-----------------|---------------|-------------------|------------------------|----------------------|-----------------|-------|
| IMG_001 | John | 2024-01-20 | Natural | N/A | Natural | Yes | Sharp | 1 | Real | 9 | Professional photo quality |
| IMG_002 | John | 2024-01-20 | Distorted | Garbled | Unnatural | No | Blurry_Edges | 9 | AI | 10 | Clear AI generation markers |
| IMG_003 | Sarah | 2024-01-21 | Natural | N/A | Natural | Yes | Clear | 1 | Real | 9 | News photo, verified source |
| IMG_004 | Sarah | 2024-01-21 | Odd_Spacing | Absent | Slightly_Off | Questionable | Melting | 8 | AI | 9 | Midjourney signature artifacts |

**Artifact_Score Scale:**
- 1-2: Clearly Real
- 3-5: Uncertain
- 6-8: Likely AI
- 9-10: Clearly AI

---

## Sheet 3: Technical Analysis Results

| Image_ID | EXIF_Present | Camera_Model | Date_Created | FotoForensics_Score | Compression_Normal | Forensic_Anomalies | Reverse_Search_Found | Found_Online_Count | TinEye_Results | Notes |
|----------|--------------|--------------|---------------|---------------------|-------------------|-------------------|---------------------|-------------------|---------------|-------|
| IMG_001 | Yes | Canon EOS R5 | 2024-01-10 | 3 | Yes | None | Yes | 15 | Unsplash original | Authentic photo |
| IMG_002 | No | N/A | Not_Available | 8 | No | Color_Banding | No | 0 | No results | No EXIF - typical AI |
| IMG_003 | Yes | Sony A7R | 2024-01-16 | 2 | Yes | None | Yes | 8 | Reuters wire photo | News source verified |
| IMG_004 | No | N/A | Not_Available | 7 | No | Edge_Artifacts | No | 0 | No results | Generated image |

**FotoForensics_Score:**
- 1-2: Normal authentic photo
- 3-5: Some anomalies (possible editing)
- 6-8: Significant anomalies (likely AI/heavily edited)
- 9-10: Clear generation artifacts

---

## Sheet 4: Tool-Based Detection Results

| Image_ID | Tool_1_Name | Tool_1_Score | Tool_1_Result | Tool_1_Confidence | Tool_2_Name | Tool_2_Score | Tool_2_Result | Tool_2_Confidence | Tool_3_Name | Tool_3_Score | Tool_3_Result | Tool_3_Confidence | Consensus_Result | Tool_Agreement |
|----------|-------------|--------------|---------------|-------------------|-------------|--------------|---------------|-------------------|-------------|--------------|---------------|-------------------|-----------------|-----------------|
| IMG_001 | HuggingFace | 0.15 | Real | 85% | FotoForensics | Real | Real | 90% | InVID | Real | Real | 88% | Real | 100% |
| IMG_002 | HuggingFace | 0.92 | AI | 92% | FotoForensics | AI | AI | 95% | InVID | AI | AI | 90% | AI | 100% |
| IMG_003 | HuggingFace | 0.18 | Real | 82% | FotoForensics | Real | Real | 87% | InVID | Real | Real | 85% | Real | 100% |
| IMG_004 | HuggingFace | 0.88 | AI | 88% | FotoForensics | AI | AI | 91% | InVID | AI | AI | 87% | AI | 100% |

---

## Sheet 5: Performance Metrics Summary

| Detection_Method | True_Positives | True_Negatives | False_Positives | False_Negatives | Accuracy_% | Precision_% | Recall_% | F1_Score | Specificity_% |
|------------------|----------------|----------------|-----------------|-----------------|-----------|-----------|---------|----------|--------------|
| Visual Analysis | 245 | 243 | 7 | 5 | 97.6% | 97.2% | 98.0% | 0.976 | 97.2% |
| FotoForensics | 238 | 246 | 4 | 12 | 97.2% | 98.3% | 95.2% | 0.967 | 98.4% |
| HuggingFace | 241 | 239 | 11 | 9 | 96.0% | 95.6% | 96.4% | 0.960 | 95.6% |
| Manual+Tool Combined | 248 | 247 | 3 | 2 | 99.0% | 98.8% | 99.2% | 0.990 | 98.8% |

---

## Sheet 6: By Model Performance

| AI_Model | Sample_Count | Avg_Artifact_Score | Detection_Accuracy_% | Easiest_Artifact | Hardest_Artifact | False_Negative_Count | Notes |
|----------|--------------|-------------------|----------------------|-----------------|-----------------|----------------------|-------|
| DALL-E 3 | 50 | 8.2 | 98% | Hand_Distortion | Texture_Quality | 1 | Most detectable |
| Midjourney | 50 | 7.8 | 96% | Edge_Melting | Overall_Coherence | 2 | Clean generation |
| Stable Diffusion | 50 | 7.5 | 94% | Color_Bleeding | Physics | 3 | Improving fast |
| ChatGPT Vision | 30 | 6.9 | 92% | Geometry | Fine_Details | 2 | Less common |
| Other | 20 | 7.6 | 95% | Varies | Varies | 1 | Mixed results |

---

## Sheet 7: Artifact Frequency Analysis

| Artifact_Type | Occurrence_in_AI_% | Occurrence_in_Real_% | Detection_Reliability_% | Most_Common_in_Model | False_Positive_Risk |
|---|---|---|---|---|---|
| Hand Distortion | 92% | 1% | 94% | DALL-E 3 | Very Low |
| Text Garbling | 87% | 2% | 96% | DALL-E 3 | Very Low |
| Eye Inconsistency | 84% | 3% | 91% | Midjourney | Low |
| Edge Melting | 79% | 4% | 89% | Midjourney | Low |
| Background Blur | 76% | 8% | 82% | Stable Diffusion | Medium |
| Color Bleeding | 71% | 6% | 79% | Stable Diffusion | Medium |
| Impossible Physics | 68% | 5% | 75% | All Models | Medium |
| Lighting Inconsistency | 62% | 7% | 70% | All Models | Medium-High |
| Texture Artifacts | 58% | 9% | 65% | Stable Diffusion | High |
| Over-Symmetry | 45% | 12% | 60% | DALL-E 3 | High |

---

## Sheet 8: Image Category Performance

| Category | Total_Images | Real_Count | AI_Count | Detection_Accuracy_% | False_Positive_% | False_Negative_% | Notes |
|----------|--------------|-----------|---------|----------------------|-----------------|-----------------|-------|
| Portrait | 100 | 50 | 50 | 98.5% | 1.2% | 0.8% | Best detection |
| Landscape | 100 | 50 | 50 | 96.2% | 2.8% | 1.0% | Good accuracy |
| Objects/Still-life | 80 | 40 | 40 | 95.8% | 3.2% | 1.0% | Moderate |
| Abstract/Artistic | 70 | 35 | 35 | 92.1% | 5.8% | 2.1% | Challenging |
| Text-Heavy | 60 | 30 | 30 | 99.2% | 0.5% | 0.3% | Easiest |
| Mixed/Complex | 90 | 45 | 45 | 93.4% | 4.2% | 2.4% | Most difficult |

---

## Sheet 9: Analysis Timeline & Progress

| Task | Planned_Start | Planned_End | Actual_Start | Actual_End | % Complete | Images_Processed | Status | Notes |
|------|---------------|-------------|--------------|------------|-----------|-----------------|--------|-------|
| Literature Review | 2024-01-01 | 2024-01-31 | 2024-01-01 | 2024-01-28 | 100% | - | Complete | Reviewed 47 papers |
| Data Collection | 2024-02-01 | 2024-02-28 | 2024-02-01 | 2024-02-20 | 100% | 500 | Complete | All verified |
| Visual Analysis | 2024-03-01 | 2024-03-31 | 2024-03-01 | In Progress | 75% | 375 | In Progress | On schedule |
| Technical Analysis | 2024-03-15 | 2024-04-15 | 2024-03-15 | In Progress | 45% | 225 | In Progress | Running slow |
| Tool Testing | 2024-04-01 | 2024-04-30 | 2024-04-05 | Pending | 0% | - | Not Started | Waiting for tool access |
| Statistical Analysis | 2024-05-01 | 2024-05-15 | Pending | Pending | 0% | - | Not Started | Pending data completion |
| Manuscript Writing | 2024-05-16 | 2024-06-30 | Pending | Pending | 0% | - | Not Started | Outline ready |

---

## Sheet 10: Quality Control Checklist

| Check_Item | Required | Completed | Date | By_Who | Notes |
|---|---|---|---|---|---|
| EXIF Data Verification | Yes | Yes | 2024-02-10 | John | All real images verified |
| Duplicate Detection | Yes | Yes | 2024-02-12 | Sarah | No duplicates found |
| Image Authenticity Review | Yes | In Progress | - | John/Sarah | 480/500 verified |
| Source Confirmation | Yes | In Progress | - | John | All sources documented |
| Format Consistency Check | Yes | Yes | 2024-02-15 | Tech | All files valid |
| Metadata Completeness | Yes | Yes | 2024-02-16 | John | 100% complete |
| Inter-Rater Reliability Test | Yes | Pending | - | John/Sarah | 10% sample to verify |
| Random Verification | Yes | In Progress | - | Independent | 5% re-analysis |
| Data Entry Accuracy | Yes | In Progress | - | Sarah | 50% verified so far |
| Reference Documentation | Yes | In Progress | - | John | 90% complete |

---

## Quick Reference Calculations

**Overall Statistics (to be auto-calculated):**

```
Total Images Analyzed: [COUNT]
- Real Images: [COUNT] ([%])
- AI-Generated Images: [COUNT] ([%])

Total Detection Methods Used: [COUNT]
Average Accuracy Across All Methods: [%]
Best Performing Method: [NAME] ([%])
Worst Performing Method: [NAME] ([%])

Overall False Positive Rate: [%]
Overall False Negative Rate: [%]

Analysis Timeline:
- Start Date: [DATE]
- Current Date: [DATE]
- Estimated Completion: [DATE]
- Percentage Complete: [%]
```

---

## Export Templates

**For Manuscript Figures:**

Table 1: Detection Tool Performance Comparison
- Auto-generate from Sheet 5

Table 2: Artifact Frequency Analysis
- Auto-generate from Sheet 7

Table 3: Category-Based Performance
- Auto-generate from Sheet 8

Figure 1: Accuracy Comparison Bar Chart
- Data from Sheet 5

Figure 2: Artifact Frequency Distribution
- Data from Sheet 7

Figure 3: Detection by Category
- Data from Sheet 8
