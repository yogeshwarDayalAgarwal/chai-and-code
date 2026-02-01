import os
import base64
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import fitz  # PyMuPDF


# ============================================================================
# CONFIGURATION
# ============================================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4-vision")
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"


# ============================================================================
# PDF PREPROCESSING
# ============================================================================
def extract_images_from_pdf(pdf_path: str) -> List[str]:
    """Extract images from PDF before passing to agent"""
    output_dir = pdf_path.replace('.pdf', '_images')
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_path = os.path.join(output_dir, f"page{page_num+1}_img{img_index+1}.{base_image['ext']}")
            
            with open(image_path, "wb") as f:
                f.write(base_image["image"])
            image_paths.append(image_path)
    
    doc.close()
    return image_paths


# ============================================================================
# INPUT SCHEMA
# ============================================================================
class ImageInput(BaseModel):
    """Input schema for morphing detection"""
    image_path: str = Field(description="Path to the medical banner image file to analyze")


# ============================================================================
# SINGLE COMPREHENSIVE TOOL
# ============================================================================
class MorphingDetectionTool(BaseTool):
    """Single tool that performs comprehensive morphing analysis"""
    
    name: str = "detect_morphing"
    description: str = """Use this tool to analyze a medical banner image for person/face morphing.
    This tool performs comprehensive forensic analysis checking:
    - Geometric alignment (proportions, perspective, positioning)
    - Blur consistency (focus patterns, sharpness mismatches)
    - Shadow and lighting (direction, intensity, consistency)
    - Edge artifacts (halos, hard edges, cut-paste signs)
    
    Input: image_path (string) - path to the banner image
    Returns: JSON with morphing detection results including regions and confidence scores"""
    
    args_schema: type[BaseModel] = ImageInput
    
    def _run(self, image_path: str) -> str:
        """Execute morphing detection analysis"""
        
        # Handle case where image_path might be JSON-wrapped (LangChain parsing issue)
        if isinstance(image_path, str) and image_path.strip().startswith('{'):
            try:
                parsed = json.loads(image_path)
                if 'image_path' in parsed:
                    image_path = parsed['image_path']
            except:
                pass  # If parsing fails, use as-is
        
        # Encode image to base64
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            return json.dumps({
                "error": f"Image file not found: {image_path}",
                "is_morphed": None,
                "morphed_regions": []
            })
        except Exception as e:
            return json.dumps({
                "error": f"Failed to read image: {str(e)}",
                "is_morphed": None,
                "morphed_regions": []
            })
        
        # Initialize Azure OpenAI
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0  # Deterministic for consistent detection
        )
        
        # Comprehensive analysis prompt
        analysis_prompt = """You are a forensic image analyst expert specializing in detecting manipulation in medical promotional materials, including person morphing AND text/graphic editing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE-BASED DETECTION PHILOSOPHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DETECTION REQUIRES MULTIPLE STRONG INDICATORS:
• Flag as morphed ONLY if you find 2+ CRITICAL indicators
• OR 1 CRITICAL + 2+ MODERATE indicators
• Single minor issue alone = NOT morphing (just normal artifacts)

Be precise and evidence-based. Distinguish between:
✓ Clear manipulation (obvious problems)
✗ Normal artifacts (JPEG compression, natural camera angles, legitimate design)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES OF CLEAR MORPHING (FLAG THESE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1 - Text Added Post-Photo:
"Banner surface is curved/folded, but text appears completely flat and 2D.
Text perspective does NOT follow the surface geometry at all.
Text looks like it was overlaid in image editing software after the photo was taken.
→ CRITICAL: Text geometry completely wrong"

EXAMPLE 2 - Person Cut-Paste:
"Person's face is sharp and well-lit, but body is blurry and differently lit.
Hard edges around person's outline with color fringing.
Person's shadow missing while other objects have shadows.
→ CRITICAL: Multiple manipulation indicators on same person"

EXAMPLE 3 - Poor Integration:
"Text has bright white halo/outline around all letters.
Text resolution much sharper than background texture.
Text shadows point wrong direction compared to other shadows.
→ CRITICAL: Text clearly added later, not part of original"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES OF AUTHENTIC (DO NOT FLAG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Text has minor JPEG artifacts but follows banner surface naturally.
Text shadows roughly match lighting direction and surface angle.
Text appears to be part of the original printed design.
→ Normal banner, don't flag"

"Some persons slightly sharper than others due to depth of field.
All persons have consistent lighting from same source.
No hard edges or halos, natural photo.
→ Normal photo quality variation, don't flag"

Analyze this medical camp banner systematically:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP-BY-STEP ANALYSIS PROCESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**STEP 1: TEXT & TYPOGRAPHY MANIPULATION**

Examine text elements and classify issues by severity:

🔴 CRITICAL TEXT INDICATORS (Strong evidence of manipulation):
• Text perspective COMPLETELY wrong (flat text on clearly curved/angled surface)
• Text obviously floating/overlaid (clear separation from background)
• Text with prominent white halos or selection marks around ALL letters
• Text shadows pointing OPPOSITE direction from other shadows
• Text that clearly doesn't follow banner surface geometry at all

🟡 MODERATE TEXT INDICATORS (Possible issues):
• Text perspective slightly off but not completely wrong
• Text edges somewhat sharper than background
• Minor shadow direction mismatch
• Text appears slightly overlaid but not obviously

🟢 MINOR ISSUES (Normal artifacts - IGNORE these):
• Small JPEG compression artifacts on text
• Natural text rendering on legitimate printed banner
• Slight shadow softness differences
• Minor camera angle effects on text appearance

**Analysis Approach:**
1. Identify ANY text issues present
2. Classify each as CRITICAL, MODERATE, or MINOR
3. Count indicators:
   - 2+ CRITICAL = Flag as morphed
   - 1 CRITICAL + 2+ MODERATE = Flag as morphed
   - Only MINOR issues = Mark as authentic

**STEP 2: PERSON/FACE MANIPULATION**

🔴 CRITICAL PERSON INDICATORS:
• Face sharp but body very blurry (or vice versa) on SAME person
• Hard edges with color fringing around person's entire outline
• Person missing shadows while other objects have clear shadows
• Face lighting completely different from body lighting
• Obvious geometric proportion problems (head too big/small for body)

🟡 MODERATE PERSON INDICATORS:
• Slight sharpness variation between face and body
• Minor edge softness issues
• Shadow slightly weaker than expected
• Slight lighting variation

🟢 MINOR ISSUES (Normal - IGNORE):
• Natural depth of field effects (background blur)
• Normal photo quality variations between persons
• Slight focus differences across image plane

**STEP 3: SHADOW & LIGHTING CONSISTENCY**

🔴 CRITICAL SHADOW/LIGHTING INDICATORS:
• Shadows pointing in COMPLETELY opposite directions
• Person clearly missing shadow when should have one
• Lighting sources obviously conflicting (person lit from left, others from right)
• Shadow darkness completely wrong for distance/lighting

🟡 MODERATE SHADOW/LIGHTING INDICATORS:
• Shadow direction slightly off (10-20 degrees)
• Shadow intensity somewhat different
• Minor lighting inconsistencies

🟢 MINOR ISSUES (Normal - IGNORE):
• Natural shadow softness variations
• Slight lighting differences from camera angle
• Normal fill light effects

**STEP 4: EDGE QUALITY & INTEGRATION**

🔴 CRITICAL EDGE INDICATORS:
• Bright halos or outlines around persons/objects
• Hard, unnatural edges with visible selection marks
• Clear layer separation (elements obviously pasted)
• Obvious compression quality differences

🟡 MODERATE EDGE INDICATORS:
• Slightly sharp edges
• Minor color fringing
• Small background blending issues

🟢 MINOR ISSUES (Normal - IGNORE):
• Natural edge sharpness from focus
• JPEG compression artifacts
• Normal photo quality characteristics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: OUTPUT MUST BE VALID JSON ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After systematic analysis, return ONLY valid JSON (no additional text):

If MORPHING DETECTED:
{
  "is_morphed": true,
  "confidence_score": "high/medium/low",
  "morphed_regions": [
    {
      "location": "exact location description (e.g., 'center banner text', 'left person face')",
      "bbox": [x1_percent, y1_percent, x2_percent, y2_percent],
      "reason": "EVIDENCE: [List 2+ specific CRITICAL/MODERATE indicators, e.g., 'CRITICAL: text perspective completely flat on curved surface, MODERATE: text shadow direction differs from other shadows']",
      "severity": "critical/moderate",
      "confidence_score": "high/medium"
    }
  ]
}

IMPORTANT: "reason" field MUST cite specific CRITICAL and MODERATE indicators found.
Do NOT use "minor" severity - only report CRITICAL or MODERATE issues.

If AUTHENTIC (NO MORPHING):
{
  "is_morphed": false,
  "morphed_regions": []
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOUNDING BOX FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bbox: [x1_percent, y1_percent, x2_percent, y2_percent]
- x1_percent: left edge as % of image width (0-100)
- y1_percent: top edge as % of image height (0-100)
- x2_percent: right edge as % of image width (0-100)
- y2_percent: bottom edge as % of image height (0-100)

Example: [30, 15, 50, 70] = box from 30% to 50% horizontally, 15% to 70% vertically

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEDICAL BANNER CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Typical medical camp banners have:
• Consistent professional photography quality
• Uniform lighting from single photo shoot
• Text that follows surface perspective naturally
• Cohesive design elements

COMMON MANIPULATION IN MEDICAL BANNERS:
• Text/dates/locations edited after printing/design
• Stock photo faces pasted onto bodies
• Text overlaid without proper perspective/integration
• Multiple elements combined from different sources
• Geometric text distortion from incorrect editing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION LOGIC (EVIDENCE-BASED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**BEFORE DECIDING**: Count your evidence:
1. List ALL issues found (text, persons, shadows, edges)
2. Classify EACH as: CRITICAL 🔴, MODERATE 🟡, or MINOR 🟢
3. Apply decision rules:

**FLAG AS MORPHED** if:
✓ 2+ CRITICAL indicators found → confidence: "high"
✓ 1 CRITICAL + 2+ MODERATE indicators → confidence: "medium"  
✓ 1 CRITICAL + 1 MODERATE (if very obvious) → confidence: "medium"

**MARK AS AUTHENTIC** if:
✓ Only MINOR issues found (normal artifacts)
✓ Only 1 MODERATE indicator without CRITICAL support
✓ No significant evidence of manipulation

**REASONING REQUIRED:**
- List specific indicators found
- Explain why each is CRITICAL, MODERATE, or MINOR
- Show your evidence count before deciding

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE LEVELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• HIGH: 3+ CRITICAL indicators OR 2 CRITICAL + strong supporting evidence
• MEDIUM: 2 CRITICAL indicators OR 1 CRITICAL + 2+ MODERATE
• LOW: 1 CRITICAL + 1 MODERATE OR questionable indicators

Only flag as morphed if confidence is at least MEDIUM.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Return ONLY valid JSON, no markdown, no additional text
2. Be precise with bounding box coordinates
3. Include ONLY regions with CRITICAL or MODERATE indicators
4. Each morphed region must cite specific evidence
5. Ignore MINOR issues (normal artifacts) - don't report them

Analyze the image now:"""

        # Send request to Azure OpenAI Vision
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            return json.dumps({
                "error": f"API call failed: {str(e)}",
                "is_morphed": None,
                "morphed_regions": []
            })


# ============================================================================
# AGENT CREATION
# ============================================================================
def create_morphing_agent() -> AgentExecutor:
    """Create agent executor for morphing detection"""
    
    # Initialize the single comprehensive tool
    tools = [MorphingDetectionTool()]
    
    # Initialize LLM for agent reasoning
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0
    )
    
    # ReAct agent prompt template
    react_prompt = PromptTemplate.from_template("""You are a medical banner verification agent that detects morphed/manipulated persons in promotional materials.

You have access to a comprehensive morphing detection tool. Use it to analyze the banner image and provide results.

TASK: Analyze the provided medical camp banner for person/face morphing and manipulation.

AVAILABLE TOOLS:
{tools}

TOOL NAMES: {tool_names}

INSTRUCTIONS:
1. Use the detect_morphing tool to analyze the image
2. The tool returns JSON with morphing detection results
3. Return that JSON as your Final Answer without modification

FORMAT:
Question: the input question/task
Thought: I need to use the morphing detection tool to analyze this banner
Action: detect_morphing
Action Input: {{"image_path": "path/to/image"}}
Observation: [tool output - JSON result]
Thought: I have the analysis results, I can provide the final answer
Final Answer: [Return the JSON from tool output]

Begin!

Question: {input}

{agent_scratchpad}""")
    
    # Create ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================
def analyze_banner(file_path: str) -> Dict[str, Any]:
    """
    Analyze medical banner using agent executor (handles both images and PDFs)
    
    Args:
        file_path: Path to the banner image file or PDF
        
    Returns:
        Dictionary containing morphing detection results
    """
    
    # Handle PDF by extracting images first
    if file_path.lower().endswith('.pdf'):
        print(f"📄 Processing PDF: {file_path}")
        print("="*70)
        
        try:
            # Extract images from PDF
            images = extract_images_from_pdf(file_path)
            print(f"✓ Extracted {len(images)} image(s) from PDF\n")
            
            # Analyze each extracted image
            all_results = []
            for idx, img_path in enumerate(images, 1):
                print(f"\n[{idx}/{len(images)}] Analyzing: {os.path.basename(img_path)}")
                result = analyze_banner(img_path)  # Recursive call with image
                if result.get("success"):
                    result["image_file"] = os.path.basename(img_path)
                all_results.append(result)
            
            # Return aggregated results
            morphed_count = sum(1 for r in all_results if r.get("success") and r.get("result", {}).get("is_morphed"))
            
            return {
                "success": True,
                "pdf_path": file_path,
                "images_analyzed": len(images),
                "morphed_images_found": morphed_count,
                "results": all_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"PDF processing failed: {str(e)}",
                "pdf_path": file_path
            }
    
    # Handle single image (original logic)
    print(f"🔍 Analyzing image: {file_path}")
    print("="*70)
    
    # Create agent
    agent_executor = create_morphing_agent()
    
    # Execute analysis
    result = agent_executor.invoke({
        "input": f"Detect person/face morphing in this medical banner: {file_path}"
    })
    
    # Parse JSON from agent output
    try:
        output_text = result["output"]
        
        # Clean output if wrapped in markdown
        if "```json" in output_text:
            output_text = output_text.split("```json")[1].split("```")[0]
        elif "```" in output_text:
            output_text = output_text.split("```")[1].split("```")[0]
        
        # Extract JSON
        if "{" in output_text:
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1
            json_str = output_text[json_start:json_end]
            parsed_result = json.loads(json_str)
            
            return {
                "success": True,
                "result": parsed_result,
                "reasoning_steps": result.get("intermediate_steps", [])
            }
        else:
            return {
                "success": False,
                "error": "No JSON found in output",
                "raw_output": output_text
            }
            
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON parsing failed: {str(e)}",
            "raw_output": result.get("output", "")
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "raw_output": result.get("output", "")
        }


def print_results(analysis_result: Dict[str, Any]):
    """Pretty print analysis results"""
    
    print("\n" + "="*70)
    print("📊 MORPHING DETECTION RESULTS")
    print("="*70)
    
    if not analysis_result.get("success"):
        print("❌ ERROR:", analysis_result.get("error"))
        print("\nRaw output:", analysis_result.get("raw_output"))
        return
    
    # Handle PDF results (multiple images)
    if "pdf_path" in analysis_result:
        print(f"📄 PDF: {analysis_result.get('pdf_path')}")
        print(f"🖼️  Images analyzed: {analysis_result.get('images_analyzed', 0)}")
        print(f"⚠️  Morphed images found: {analysis_result.get('morphed_images_found', 0)}")
        print("\n" + "-"*70)
        
        for idx, img_result in enumerate(analysis_result.get('results', []), 1):
            print(f"\n[Image {idx}] {img_result.get('image_file', 'Unknown')}")
            if img_result.get("success") and img_result.get("result"):
                result = img_result["result"]
                if result.get("is_morphed"):
                    print(f"  ⚠️  MORPHED - Confidence: {result.get('confidence_score', 'unknown')}")
                    print(f"  📍 Regions: {len(result.get('morphed_regions', []))}")
                else:
                    print(f"  ✅ AUTHENTIC")
            else:
                print(f"  ❌ ERROR: {img_result.get('error', 'Unknown error')}")
        
        print("\n" + "="*70)
        return
    
    # Handle single image result
    result = analysis_result["result"]
    
    if result.get("is_morphed"):
        print("⚠️  VERDICT: MORPHED/MANIPULATED")
        print(f"🎯 Confidence: {result.get('confidence_score', 'unknown').upper()}")
        print(f"📍 Regions detected: {len(result.get('morphed_regions', []))}")
        
        print("\n" + "-"*70)
        print("SUSPICIOUS REGIONS:")
        print("-"*70)
        
        for idx, region in enumerate(result.get('morphed_regions', []), 1):
            print(f"\n[{idx}] Location: {region.get('location')}")
            print(f"    Bounding Box: {region.get('bbox')}")
            print(f"    Severity: {region.get('severity', 'unknown').upper()}")
            print(f"    Confidence: {region.get('confidence_score', 'unknown')}")
            print(f"    Reason: {region.get('reason')}")
    else:
        print("✅ VERDICT: AUTHENTIC")
        print("No morphing or manipulation detected")
    
    print("\n" + "="*70)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Check if image/PDF path provided
    if len(sys.argv) > 1:
        banner_path = sys.argv[1]
    else:
        # Default example path
        banner_path = "medical_camp_banner.jpg"
        print(f"ℹ️  No file provided, using default: {banner_path}")
        print(f"   Usage: python agent.py <path_to_image_or_pdf>")
        print()
    
    # Check environment variables
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        print("❌ ERROR: Azure OpenAI credentials not found!")
        print("\nPlease set environment variables:")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_KEY")
        print("  - AZURE_OPENAI_DEPLOYMENT (optional, defaults to 'gpt-4-vision')")
        sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(banner_path):
        print(f"❌ ERROR: File not found: {banner_path}")
        sys.exit(1)
    
    # Run analysis
    try:
        analysis_result = analyze_banner(banner_path)
        print_results(analysis_result)
        
        # Optionally save results to JSON file
        output_file = banner_path.replace(".", "_analysis.") + ".json"
        if analysis_result.get("success"):
            with open(output_file, "w") as f:
                # Handle both single image and PDF results
                if "pdf_path" in analysis_result:
                    # PDF result - save entire structure
                    json.dump(analysis_result, f, indent=2)
                else:
                    # Single image result - save just the result
                    json.dump(analysis_result["result"], f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
