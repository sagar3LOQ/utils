import sys, os, time, re

# For PDF to text conversion:
import textract

# To identify file type:
import magic # Needs libmagic1 installed.

# For parsing CV using a grammar definition:
import pyparsing

# For doc, docx and odt to text conversions:
import docx2txt



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
 #       randomStr = 1
        self.label = label
#        self.userTempDir = self.outDir + os.path.sep + "tmp_" + randomStr.__str__()


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
            self._convert_pdf_to_text(index)
        elif self.cvFormat == 'docx':
            self._convert_docx_to_text(index)
        elif self.cvFormat == 'doc':
            self._convert_doc_to_text(index)
        elif self.cvFormat == 'rtf':    
            self._convert_rtf_to_text(index)
        elif self.cvFormat == 'txt':
            print "Unrecognised format : pass decoding"
        else:
            print "Unrecognised format : pass decoding"
        return(self.cvTextFile)


    def _convert_pdf_to_text(self,index):
      #  print "processing pdfs"
        input_pdf = self.cvFile


        outputPath = self.outDir
        inputPath = os.getcwd()
        if os.path.exists(input_pdf):
            inputPath = os.path.dirname(input_pdf)
        input_filename = os.path.basename(input_pdf)
        input_parts = input_filename.split(".")
        input_parts.pop()
        randomStr = int(time.time())
        output_filename = outputPath + os.path.sep + "".join(str(index)) + "_"+str(self.label)   + r".txt"
        self.cvTextFile = output_filename

        output_filename = output_filename.replace (" ", "_")
        outfp = open(output_filename, 'w')

        text = textract.process(input_pdf)
 
        outfp.write(text)

        outfp.close()
        return (0)

    def _convert_rtf_to_text(self,index):
   #     print "processing rtf"
        input_pdf = self.cvFile

        outputPath = self.outDir
        inputPath = os.getcwd()
        if os.path.exists(input_pdf):
            inputPath = os.path.dirname(input_pdf)
        input_filename = os.path.basename(input_pdf)
        input_parts = input_filename.split(".")
        input_parts.pop()
        randomStr = int(time.time())
        output_filename = outputPath + os.path.sep + "".join(str(index)) + "_"+str(self.label)   + r".txt"
        self.cvTextFile = output_filename

        outfp = open(output_filename, 'w')
        text = textract.process(input_pdf)
        outfp.write(text)

        outfp.close()
        return (0)
    
    def _convert_doc_to_text(self,index, password=None):

        input_doc = self.cvFile
        outputPath = self.outDir

        inputPath = os.getcwd()
        if os.path.exists(input_doc):
            inputPath = os.path.dirname(input_doc)

        input_filename = os.path.basename(input_doc)
        input_parts = input_filename.split(".")
        print "testing"
        output_filename = outputPath + os.path.sep + "".join(str(index)) + "_"+str(self.label)  + r".txt"        #output_filename = output_filename.replace (" ", "_")

        self.cvTextFile = output_filename
        cmd = 'catdoc "%s" > "%s"'%(self.cvFile, self.cvTextFile) # Dangerous!!! Why not use 'subprocess'?
        os.system(cmd)
        return(0)



    def _convert_docx_to_text(self,index, password=None):

#        print "Decoding docx file"
        input_docx = self.cvFile
        outputPath = self.outDir
        inputPath = os.getcwd()
        if os.path.exists(input_docx):
            inputPath = os.path.dirname(input_docx)
        input_filename = os.path.basename(input_docx)
        input_parts = input_filename.split(".")
        input_parts.pop()
        randomStr = int(time.time())
        output_filename = outputPath + os.path.sep + "".join(str(index)) + "_"+str(self.label)   + r".txt"

 #       print "writing output to {0}".format(output_filename)
        text = docx2txt.process(input_docx)
        fw = open(output_filename, "w")

        fw.write(text.encode('utf-8'))
        fw.close()
        return(0)


def convertFiles(in_dir,label,out_dir):
    index = 0
    for root, dirs, files in os.walk(in_dir):
#    for cvfile in os.listdir(in_dir):
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
                cvparser.preprocess(index)
            except:
                print "Conversion Failed :("
                sys.exit(1)

def genLabel(strg):
    return strg.split("/")[-1]
	

if __name__ == '__main__':
#    cvfile = sys.argv[1]

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


