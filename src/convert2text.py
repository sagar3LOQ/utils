import sys, os, time, re

# For PDF to text conversion:
import textract

# To identify file type:
import magic # Needs libmagic1 installed.

# For parsing CV using a grammar definition:
import pyparsing

# For doc, docx and odt to text conversions:
import docx2txt

import hashlib

def getMD5HashDigest(text):
    return hashlib.md5(text).hexdigest()

def getOutName(md5Str,label):


    strg = str(md5Str) + "_" + label + ".dat"
    return strg

class CVParser(object):
    # Some commonly used regex patterns:
    def __init__(self, cvfile,label,outDir, password=None):
        self.errorMsg = None
        self.cvFile = cvfile # Input CV file - can be in any of the following formats: pdf, doc, docx, rtf, txt, odt.
        if not os.path.exists(self.cvFile) or not os.path.isfile(self.cvFile):
            self.errorMsg = "The input file '%s' doesn't exist\n"%self.cvFile
            return None
        self.cvFormat = None # Can be any of the following: pdf, doc, docx, rtf, txt, odt.
        self.cvFilePasswd = password # password for the input CV file (if one exists).
        self.outDir =  outDir
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)
        self.cvTextFile = None

        self.label = label


    """
    Identify the format of the input file.
    """
    def _checkFormat(self):
        mime = magic.Magic(mime=True)
        filetype = mime.from_file(self.cvFile)
        fileparts = self.cvFile.split(".")
        ext = fileparts.pop()
        ext = ext.lower()
        primaryEnc, secondaryEnc = filetype.split("/")

        if ext == 'pdf' or ext == 'doc' or ext == 'docx' or ext =='rtf':
            self.cvFormat = ext

            return(True)
        elif ext == 'text' :
            self.cvFormat = 'txt'

            return(True)
            
        else:
             return(True)
            

    """
    This method will identify the type of the input file and dispatch the file to the appropriate convertor method.
    """
    def preprocess(self,index):
        self._checkFormat()
        if self.errorMsg is not None:
            print self.errorMsg
            sys.exit(0)
        if self.cvFormat == 'pdf':
            return self._convert_pdf_to_text(index)
        elif self.cvFormat == 'docx':
            return self._convert_docx_to_text(index)
        elif self.cvFormat == 'doc':
            return self._convert_doc_to_text(index)
        elif self.cvFormat == 'rtf':    
            return self._convert_rtf_to_text(index)
        elif self.cvFormat == 'txt':
            print "Unrecognised format : pass decoding"
        else:
            print "Unrecognised format : pass decoding"
        return(0)


    def _convert_pdf_to_text(self,index):
      #  print "processing pdfs"
        input_pdf = self.cvFile


        inputPath = os.getcwd()
        if os.path.exists(input_pdf):
            inputPath = os.path.dirname(input_pdf)
        input_filename = os.path.basename(input_pdf)
        input_parts = input_filename.split(".")
        input_parts.pop()

        text = textract.process(input_pdf)
 
        return text

    def _convert_rtf_to_text(self,index):
   #     print "processing rtf"
        input_pdf = self.cvFile

        inputPath = os.getcwd()
        if os.path.exists(input_pdf):
            inputPath = os.path.dirname(input_pdf)
        input_filename = os.path.basename(input_pdf)
        input_parts = input_filename.split(".")
        input_parts.pop()

        text = textract.process(input_pdf)
        return text
    
    def _convert_doc_to_text(self,index, password=None):

        input_doc = self.cvFile

        inputPath = os.getcwd()
        if os.path.exists(input_doc):
            inputPath = os.path.dirname(input_doc)

        input_filename = os.path.basename(input_doc)
        input_parts = input_filename.split(".")


        cmd = 'catdoc "%s"'%(self.cvFile) # Dangerous!!! Why not use 'subprocess'?
        text = os.popen(cmd).read()

        return text



    def _convert_docx_to_text(self,index, password=None):

#        print "Decoding docx file"
        input_docx = self.cvFile

        inputPath = os.getcwd()
        if os.path.exists(input_docx):
            inputPath = os.path.dirname(input_docx)
        input_filename = os.path.basename(input_docx)
        input_parts = input_filename.split(".")
        input_parts.pop()
        text = docx2txt.process(input_docx)

        return text.encode('utf-8')


def convertFiles(in_dir,label,out_dir):
    index = 0
    for root, dirs, files in os.walk(in_dir):


        for file in files:
            index += 1
            cvfile = os.path.join(root, file)
            print "Processing :: " + cvfile
            in_fname = genLabel(cvfile)
            out_fname = ""
            cvparser = CVParser(cvfile,label,out_dir)
            if cvparser.errorMsg:
                print cvparser.errorMsg
                sys.exit(0)
            try:
                text = cvparser.preprocess(index)
		
                md5_str = getMD5HashDigest(text)
		
                outFname = getOutName(md5_str,label)
		print "preprocess"
                outPath = out_dir + "/" + outFname
                fw = open(outPath, "w")
                fw.write(text)
                fw.close()
            except:
                print "Conversion Failed :("
                sys.exit(1)

def genLabel(strg):
    return strg.split("/")[-1]
	

if __name__ == '__main__':

    accept_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/accept"
    accept_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/accept"
    label = genLabel(accept_dir)
    convertFiles(accept_dir,label,accept_out)

    reject_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/reject"
    reject_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/reject"
    label = genLabel(reject_dir)
    convertFiles(reject_dir,label,reject_out)


    predict_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/predict"
    predict_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/predict"
    label = genLabel(predict_dir)
    convertFiles(predict_dir,label,predict_out)


