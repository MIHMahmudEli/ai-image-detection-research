# Research Guide: Detecting AI-Generated vs Real Images

## 1. VISUAL ARTIFACTS & ANOMALIES

### What to Look For:

**Hands & Fingers**
- Unusual number of fingers or digits
- Incorrectly shaped hands
- Unnatural finger positioning
- Missing or duplicated fingers

**Eyes & Facial Features**
- Asymmetrical eyes
- Mismatched iris colors
- Unusual pupil shapes
- Inconsistent facial proportions

**Text & Symbols**
- Garbled or nonsensical text
- Misspelled words that look plausible
- Inconsistent font styles
- Illegible symbols or labels

**Hair & Fur**
- Unnatural texture patterns
- Inconsistent strand directions
- Odd color blending
- Missing or merged hair sections

**Edges & Boundaries**
- Soft or blurred edges where they should be sharp
- Unnatural transitions between objects
- Missing depth boundaries
- Incorrect shadow casting

**Background Inconsistencies**
- Blurred or unrealistic backgrounds
- Missing details in secondary objects
- Impossible geometry
- Lighting that doesn't match foreground

---

## 2. TECHNICAL ANALYSIS METHODS

### Method 1: Metadata Inspection
**Steps:**
1. Right-click on the image file
2. Check "Properties" or "Get Info"
3. Look for:
   - Creation/modification timestamps
   - Camera model information
   - EXIF data (if present or absent)
   - GPS coordinates
   - Software that created the file

**What to Notice:**
- Real photos have detailed EXIF data
- AI images often lack or have fake EXIF data
- Timestamps may be suspiciously recent

### Method 2: Reverse Image Search
**Process:**
1. Use Google Images, TinEye, or Bing Images
2. Upload the image in question
3. Check if it appears elsewhere online
4. Look for original sources and dates
5. Verify the image history

**Indicators:**
- No results = possibly AI-generated
- Multiple sources with different contexts = suspicious
- Original source is recent = may be new AI image

### Method 3: Pixel-Level Analysis
**Tools to Use:**
- Adobe Photoshop (clone detection tools)
- Forensic Photoshop plugins
- Error Level Analysis (ELA) tools
- Noise consistency analyzers

**What to Check:**
- Compression artifacts
- Color channel consistency
- Noise patterns (AI images have uniform noise)
- Frequency domain anomalies

---

## 3. AI DETECTION TOOLS

### Recommended Software:

**Web-Based Tools:**
1. **Hugging Face AI Image Detector**
   - Website: huggingface.co
   - Detects images from Stable Diffusion, DALL-E, Midjourney
   - Accuracy: 70-90%

2. **Sensity**
   - Specialized in deepfake detection
   - Analyzes facial authenticity
   - Good for detecting manipulated faces

3. **Reality Defender**
   - Cloud-based detection
   - Multiple detection models
   - Batch processing available

**Desktop Tools:**
1. FFForensics
   - Open-source forensic toolkit
   - Detects manipulations and AI generation
   - Requires technical setup

2. Forgery Detection Toolkit
   - Specialized analysis software
   - Free for research purposes

---

## 4. BEHAVIORAL RED FLAGS

**When Sharing/Publishing:**
- Refusal to provide image source
- Vague origin story
- Claims of "found online" without link
- Suspicious timing with news events
- Perfect composition (too good to be true)
- Unusual distribution pattern

**Image Properties:**
- Overly polished appearance
- Uniform lighting (unnaturally perfect)
- Generic or common scenarios
- Faces that seem "off" but realistic
- Perfect resolution consistency

---

## 5. FORENSIC ANALYSIS CHECKLIST

Use this systematic approach:

### Visual Inspection
- [ ] Examine hands and fingers carefully
- [ ] Check facial features for asymmetries
- [ ] Read any text in the image
- [ ] Look at hair/fur texture patterns
- [ ] Check shadow and light consistency
- [ ] Examine background details

### Technical Inspection
- [ ] Extract EXIF data
- [ ] Run reverse image search
- [ ] Analyze file size vs. resolution
- [ ] Check metadata timestamps
- [ ] Look for compression inconsistencies

### Tool-Based Analysis
- [ ] Use AI detection web tools
- [ ] Run forensic software analysis
- [ ] Check pixel-level anomalies
- [ ] Analyze color channels
- [ ] Test with multiple detection tools

### Source Verification
- [ ] Find original source if possible
- [ ] Check publication credibility
- [ ] Verify photographer/creator credentials
- [ ] Look for corroborating images
- [ ] Check timeline consistency

---

## 6. CURRENT AI GENERATION PATTERNS

### Common AI Generator Artifacts:

**DALL-E 3**
- Generally higher quality
- Still struggles with hands
- Occasional text errors
- Unusual jewelry/patterns sometimes

**Midjourney**
- Very clean, polished look
- Inconsistent minor details
- Perfect lighting (suspicious)
- Sometimes overly artistic

**Stable Diffusion**
- More visible artifacts
- Strange symmetries
- Color bleeding at edges
- Lower overall quality

**Adobe Firefly**
- Seamless integration
- High quality
- Natural backgrounds
- Difficult to detect

---

## 7. LIMITATIONS OF DETECTION

**What Detection Methods Cannot Do:**
- 100% certainty (always some margin of error)
- Detect new/advanced AI models immediately
- Definitively prove an image is real
- Detect small, subtle manipulations
- Keep pace with improving AI models

**Why Detection is Challenging:**
- AI models improve rapidly
- Detection tools lag behind generation
- Some fake images look more real than real photos
- Different generators leave different signatures
- Multiple tools may give conflicting results

---

## 8. RESEARCH METHODOLOGY

### Recommended Approach:

**Step 1: Initial Assessment**
- Visual inspection (5-10 minutes)
- Note any obvious anomalies
- Document your observations

**Step 2: Automated Analysis**
- Test with 3-5 different detection tools
- Compare results
- Note confidence scores

**Step 3: Forensic Examination**
- Extract and analyze metadata
- Perform reverse image search
- Check pixel-level anomalies

**Step 4: Source Verification**
- Find original source if possible
- Verify publication details
- Check for similar images

**Step 5: Documentation**
- Record all findings
- Note tool results and confidence levels
- Provide reasoning for your conclusion
- Acknowledge uncertainty

---

## 9. KEY STATISTICS & FACTS

- **Detection Accuracy**: Most tools are 70-95% accurate on recent models
- **Lag Time**: Detection tools lag 3-6 months behind new AI models
- **False Positives**: Real images sometimes flagged as AI (2-5% of cases)
- **Rapidly Improving**: New detectors emerge monthly
- **Arms Race**: As detection improves, generation improvements follow

---

## 10. RECOMMENDED RESOURCES

**Research Papers:**
- "Detecting AI-Generated Images: A Survey" - Academic databases
- Studies on AI forensics - arXiv.org
- Deepfake detection research

**Organizations:**
- UC Berkeley's AI transparency initiative
- MIT Media Lab forensic research
- Adobe's Content Authenticity Initiative (CAI)

**Online Communities:**
- r/MediaForensics on Reddit
- AI Detection Discord communities
- Academic forums on image forensics

---

## CONCLUSION

Detecting AI-generated images requires a **multi-method approach**. No single method is foolproof. The most reliable approach combines:
1. Visual inspection + critical thinking
2. Multiple detection tools
3. Forensic analysis
4. Source verification

As AI improves, detection becomes harder, making corroboration with trustworthy sources increasingly important.
