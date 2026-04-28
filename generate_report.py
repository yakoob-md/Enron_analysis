import os
import glob
from fpdf import FPDF
from PIL import Image

def get_title_and_caption(filename, pipeline):
    basename = os.path.basename(filename).replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    
    title_parts = basename.split('_')
    
    title = f"{pipeline.capitalize()} Pipeline: "
    caption = ""
    
    if 'roc' in basename:
        title += "ROC Curve"
        caption = "This graph shows the Receiver Operating Characteristic (ROC) curve. It visualizes the trade-off between the true positive rate and false positive rate. A higher Area Under the Curve (AUC) indicates better discrimination between classes."
    elif 'pr' in basename:
        title += "Precision-Recall Curve"
        caption = "This graph presents the Precision-Recall (PR) curve, which is particularly useful for evaluating models on imbalanced datasets. It highlights the trade-off between precision (positive predictive value) and recall (sensitivity)."
    elif 'f1' in basename or 'threshold' in basename:
        title += "F1 Score & Threshold Optimization"
        caption = "This plot illustrates how the F1 score and potentially other metrics vary with different classification thresholds. It helps in identifying the optimal threshold for classifying positive and negative instances."
    elif 'cm' in basename or 'confusion' in basename:
        title += "Confusion Matrix"
        caption = "The confusion matrix provides a detailed breakdown of correct and incorrect predictions made by the model, categorized by actual and predicted classes."
    elif 'learning' in basename or 'training' in basename:
        title += "Learning Curves"
        caption = "This graph shows the model's learning progress over epochs or dataset sizes. It compares training and validation performance, indicating whether the model is overfitting, underfitting, or generalizing well."
    elif 'pred_probab' in basename:
        title += "Prediction Probability Distribution"
        caption = "This plot displays the distribution of the model's predicted probabilities. It helps in understanding the confidence of the model's predictions and how well separated the classes are."
    elif 'comparison' in basename:
        title += "Model Comparison Summary"
        caption = "This table/chart summarizes the comparative performance metrics across different models evaluated in this pipeline."
    else:
        title += " ".join(p.capitalize() for p in title_parts)
        caption = "This visualization provides insights into the experimental outputs and model performance metrics."
        
    model_name = [p for p in title_parts if p.lower() in ['bert', 'bilstm', 'lr', 'rf', 'svm', 'xgb', 'multi']]
    if model_name:
        title += f" for {' '.join(m.upper() for m in model_name)}"
        
    return title, caption

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.toc_entries = []

    def header(self):
        if self.page_no() > 2: # Skip title page and TOC
            self.set_font("Times", "I", 10)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, "Project Results and Visual Analysis", align="R")
            self.ln(10)

    def footer(self):
        if self.page_no() > 1: # Skip title page
            self.set_y(-15)
            self.set_font("Times", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_title_page(self):
        self.add_page()
        self.set_font("Times", "B", 24)
        self.ln(60)
        self.cell(0, 20, "Project Results and Visual Analysis", align="C")
        self.ln(15)
        self.set_font("Times", "I", 16)
        self.cell(0, 10, "Compiled Graphs and Experimental Outputs", align="C")
        self.ln(30)
        self.set_font("Times", "", 12)
        self.cell(0, 10, "Generated automatically for report inclusion", align="C")
        
    def add_toc(self):
        self.add_page()
        self.set_font("Times", "B", 18)
        self.cell(0, 10, "Table of Contents", align="L")
        self.ln(15)
        
        self.set_font("Times", "", 12)
        for i, (title, page) in enumerate(self.toc_entries, 1):
            self.cell(160, 10, f"Figure {i}: {title}")
            self.cell(0, 10, str(page), align="R")
            self.ln(8)
            
    def add_figure(self, img_path, title, caption, fig_num):
        self.add_page()
        
        # Record for TOC
        self.toc_entries.append((title, self.page_no()))
        
        # Title
        self.set_font("Times", "B", 14)
        self.cell(0, 10, f"Figure {fig_num}: {title}", align="L")
        self.ln(15)
        
        # Image
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                aspect = h / w
                
                # Max dimensions
                max_w = 180
                max_h = 200
                
                img_w = min(max_w, 180)
                img_h = img_w * aspect
                
                if img_h > max_h:
                    img_h = max_h
                    img_w = img_h / aspect
                    
                x = (210 - img_w) / 2 # Center
                
                self.image(img_path, x=x, w=img_w, h=img_h)
                self.ln(img_h + 10)
        except Exception as e:
            self.set_font("Times", "", 10)
            self.cell(0, 10, f"[Error loading image: {e}]")
            self.ln(15)
            
        # Caption
        self.set_font("Times", "I", 11)
        self.multi_cell(0, 6, caption)
        self.ln(10)

def main():
    dirs = [
        (r'c:\Users\dabaa\OneDrive\Desktop\dektop_content\NLP3\binary_pipeline\results', 'Binary'),
        (r'c:\Users\dabaa\OneDrive\Desktop\dektop_content\NLP3\multiclass_pipeline\results', 'Multiclass')
    ]
    
    images = []
    for d, pipeline in dirs:
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append((os.path.join(root, f), pipeline))
                    
    # Generate temporary PDF to get page numbers, then actual PDF
    # FPDF2 allows TOC generation, but a simpler 2-pass approach is robust if needed, 
    # but we can just put TOC at the end and move it to the front, or just accept TOC at the end.
    # Alternatively, FPDF2 `insert_toc_placeholder` is the modern way. Let's try basic manual TOC generation at the end and the user can just use it, or we do a 2-pass.
    # To keep it simple: We'll put the TOC at the beginning but we need the page numbers. 
    # Actually, we can generate the document twice.
    
    def build_pdf(add_toc=False, toc_entries=[]):
        pdf = PDFReport()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_title_page()
        
        if add_toc:
            pdf.add_page()
            pdf.set_font("Times", "B", 18)
            pdf.cell(0, 10, "Table of Contents", align="L")
            pdf.ln(15)
            pdf.set_font("Times", "", 12)
            for i, (title, page) in enumerate(toc_entries, 1):
                pdf.cell(160, 10, f"Figure {i}: {title[:60] + '...' if len(title) > 60 else title}")
                pdf.cell(0, 10, str(page), align="R")
                pdf.ln(8)
                
        pdf.toc_entries = [] # Reset
        
        fig_num = 1
        for img_path, pipeline in images:
            title, caption = get_title_and_caption(img_path, pipeline)
            pdf.add_figure(img_path, title, caption, fig_num)
            fig_num += 1
            
        return pdf

    # Pass 1: get TOC
    pdf_dummy = build_pdf()
    
    # Pass 2: actual PDF
    # Adjust page numbers for TOC length. Let's say TOC takes N pages.
    # 35 items roughly takes 1-2 pages.
    entries_per_page = 25
    toc_pages = (len(images) // entries_per_page) + 1
    
    adjusted_toc = [(t, p + toc_pages) for t, p in pdf_dummy.toc_entries]
    
    final_pdf = build_pdf(add_toc=True, toc_entries=adjusted_toc)
    
    output_path = r'c:\Users\dabaa\OneDrive\Desktop\dektop_content\NLP3\Project_Results_Compilation.pdf'
    final_pdf.output(output_path)
    print(f"Successfully generated PDF at {output_path}")

if __name__ == "__main__":
    main()
