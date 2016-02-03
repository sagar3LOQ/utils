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

tikkaPath = '/home/viswanath/Downloads/tika-app-1.11.jar'

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
            return self._text_process()
        else:
            print "Unrecognised format : pass decoding"
        return(0)

    def _text_process(self):
        return open(self.cvFile).read()


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


        cmd = 'catdoc "%s"'%(self.cvFile)
        text = os.popen(cmd).read()

        if text == '':
            print "Java Doc conversion %s"%(input_filename)
            cmdJava = 'java -jar ' + tikkaPath +' --text "%s"'%(self.cvFile)        
            text = os.popen(cmdJava).read()

        return text



    def _convert_docx_to_text(self,index, password=None):

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

    print in_dir

    for file in os.listdir(in_dir):
        
        file = os.path.join(in_dir, file)

        if os.path.isfile(file):
            index += 1

            cvfile = file
            print "Processing :: " + file
            
           
            cvparser = CVParser(cvfile,label,out_dir)
            if cvparser.errorMsg:
                print cvparser.errorMsg
                sys.exit(0)
            try:
                text = cvparser.preprocess(index)
		
                md5_str = getMD5HashDigest(text)
		
                outFname = getOutName(md5_str,label)
	
                outPath = out_dir + "/" + outFname
		if os.path.isfile(outPath):
		    print "File exist ... ! Text Discarded :/" 
		    continue 
                fw = open(outPath, "w")
                fw.write(text)
                fw.close()
            except:
                print "Conversion Failed :("
                sys.exit(1)

def genLabel(strg):
    return strg.split("/")[-1]
	

def convertDirFiles(dirIn,dirOut):
    label = genLabel(dirIn)
    print label
    convertFiles(dirIn,label,dirOut)

if __name__ == '__main__':

    print "Started code"
 #   accept_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/accept"
 #   accept_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/accept"
 #   convertDirFiles(accept_dir,accept_out)


  #  reject_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/reject"
  #  reject_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/reject"
  #  convertDirFiles(reject_dir,reject_out)


    predict_dir = "/home/viswanath/Downloads/reso"
    predict_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/predict"
    convertDirFiles(predict_dir,predict_out)


