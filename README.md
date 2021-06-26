# Diagram - Template Matcher

This repository contains a template matcher, which takes templates from a folder and matches them with pdfs of electrical circuit diagrams. The labels next to these templates are also recognized, using both pdfminer for text extraction and Google Tesseract's OCR library for character recognition. The program itself is a live server which runs on localhost and provides a small user interface to load in diagrams and templates.

# USAGE

To start using this repository, execute the following steps:

1. Clone this repository via git clone https://github.com/AmbitiousBoys20/DiagramAnalysis.git or download the repository.
2. Install python (3.6+)
3. Install the requirements, through pip install -r requirements.txt
4. Install the tesseract-OCR library through sudo apt-get install tesseract-ocr
5. Store templates in a folder, and .pdf files of circuit diagrams in static/diagrams. Store them in this folder, otherwise they won't load into the application!
6. Run the program with python TemplateMatcher.py. This should start a server on the localhost http://127.0.0.1:5000 and open your default browser on this specific page.

<p align="center">
  <img src="https://github.com/AmbitiousBoys20/DiagramAnalysis/blob/master/meestergif.gif" />
</p>

In the textbox on the left, fill in the template folder with the desired templates. Click the diagram which needs analysis. Then tweak the threshold and scale hyperparameters. Threshold determines how strong the template needs to correlate with the found image to determine a match. The scale range from minimum to maximum determines on which scales (in terms of size) the template has to be resized. The number of scales just determines how often the template should be resized to a different scale.

For diagram 1 use a threshold of 0.8<br>
For diagram 2 use a threshold of 0.8<br>
For diagram 3 use a threshhold of 0.65

# DOCUMENTATION

## Class:

Instantiate a new template matching class with TemplateMatcher(...). Takes in the following arguments:
  - template_dir: The directory where the templates should be stored that are being matched with the circuit diagram.
  - diagram: The path of the stored diagram
  - detection_rate: The threshold of the matches of OpenCV's matchTemplates function. A detection rate of 0.8 is the default. The threshold is responsible for how sensitive the template matcher finds matches on the diagram.
  - scale_min: There's an option to scale the templates. scale_min is responsible for the smallest size a template can be matched on. i.e. a scale of 0.25 will shrink the template down to 0.25 times its' original size, and tries to match it with the diagram. Default is 1.0.
  - scale_max: Maximum scale for the template to be matched on. Default is 1.0.
  - scale_num: The number of scales the algorithm should search on in between scale_min and scale_max. If for example a scale_num of 50 is chosen with a scale_min of 0.5 and a scale_max of 2.0, the template is resized 50 times between 0.5 and 2.0; and then matched on the diagram. This functionality is slow, since it is computationally heavy! Default is 1.

## Functions:

**match_templates**():
  - This is the heart of the class. It takes the diagram and template folder of the TemplateMatcher class and loads both into python. It then uses OpenCV's Normalized Cross Coefficient template matcher for each template. It also looks on different scale if the user has indicated this in the class instantiation. The templates find multiple matches on the same spot. To solve this, clusters of matches are created based on euclidean distance, and these matches are then analyzed on their match strength. The higher the match strength, the better the template matches on that location. Therefore the algorithm chooses the match in the cluster with the highest match strength. The best matches are then stored in a numpy array with their location, width, height and belonging template. The center of the template's bounding box is also stored to easily find the distance to the labels later on. The rectangles are then placed around the templates with place_rect. After this, the function tries to extract text from the pdf with label_extraction. If this fails, it does Optical Character Recognition with the pytesseract library. For both methods, it finds the closest label to the found template match for a particular symbol. It then places rectangles (in a different color) around the labels as well. Lastly, match_templates creates a new pdf file 'out.pdf' which contains the rectangles around the found templates and labels. It returns:
    - The unique names of the templates
    - The location of the templates
    - The number of occurrences for a template
    - The labels corresponding to the templates
    - A list of the found templates

**place_rect**():
  - This function places a rectangle on the diagram corresponding to the given bounding box and color.
  - args
    - bbox: bounding box of the rectangle area and size on the diagram
    - color: BGR color for the rectangle

**label_extraction**():
  - This function uses the pdfminer module to read the bytes of a pdf file. The pdf file is then analyzed with a parser. If a pdf has extractable text, then pdfminer is able to do so. It returns the location of the text as well as the text itself. The label_extraction function creates a bounding box relative to the pdf and stores the label as well.
    - **parse_obj**() is a child function of label_extraction. This parses every text label found individually by the label_extraction function.

**label_OCR**():
 - This function uses Google's C++ OCR Tesseract with a python wrapper called pytesseract. It searches a small area for a label. It then, just like label_extraction, creates a bounding box for the label relative to the pdf. Furthermore, it also returns the text. Pytesseract is unreliable, and it is strongly advised to use pdfs that have selectable and/or extractable text.
 - args
   - img_section: A 70 by 70 image area around a found template. This is where tesseract will search for a label.
   - loc: the location of the area relative to the diagram pdf.

**index_page**():
 - Loads the index page and loads the pdfs in the static/diagrams folder. 

**template_POST**():
 - Handles the request from the front-end to match the selected diagram with the selected template folder. It returns a json dict object with the found templates, counts, locations, etc. 
