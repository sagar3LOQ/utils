import sys, os, time, re

# For PDF to text conversion:
import textract

# To identify file type:
import magic # Needs libmagic1 installed.


import html2text

# For doc, docx and odt to text conversions:
import docx2txt

import hashlib

import tarfile

from pyunpack import Archive

from cleanData import cleanse_data

tikkaPath = '/home/viswanath/Downloads/tika-app-1.11.jar'

def getMD5HashDigest(text):
    return hashlib.md5(text).hexdigest()

def getOutName(md5Str,label):

    strg = str(md5Str) + "_" + label + ".dat"
    return strg

def cleanData(text):

##
	##  Remove all non relevent symbols and get the text
	## that can be used to clean our data with noise
##

	text = re.sub(r'[^\x00-\x7F]+',' ', text)
	text = re.sub(r"[\r\n]+[\s\t]+",'\n',text)	
	text = re.sub(r"\.[\s\t\n]+",'\n',text)	
	return text

## Class for converting CV's doc and  stored into text files 


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
        self.cvTextFile = None

        self.label = label
        if(outDir == ''): return
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)
        return

    
  ####  Identify the format of the input file.
    
    def _checkFormat(self):
        mime = magic.Magic(mime=True)
        filetype = mime.from_file(self.cvFile)
        fileparts = self.cvFile.split(".")
        ext = fileparts.pop()
        ext = ext.lower()
        primaryEnc, secondaryEnc = filetype.split("/")

        if ext == 'pdf' or ext == 'doc' or ext == 'docx' or ext =='rtf' or ext == 'html' or ext == 'htm':
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
        elif self.cvFormat == 'html':    
            return self._convert_htm_to_text()
        elif self.cvFormat == 'htm':   
            print "html conversion"
            return self._convert_htm_to_text()
        elif self.cvFormat == 'txt':
            return self._text_process()
        else:
            print "Unrecognised format : pass decoding"
        return(0)

    def _text_process(self):

        return open(self.cvFile).read()


### convert html to text
    def _convert_htm_to_text(self):

        htmlText =  open(self.cvFile).read()
        print htmlText
        s = html2text.html2text(htmlText.decode('utf-8'))
        print "html text"
        print s
        return s


###convert PDFs to text
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


### convert RTFs to text
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
  

 ####  convert DOCs to text   
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


#### convert DOCX files to text

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

#### A class defined for uncompressing the compressed files

class decompressFiles:

    def __init__(self,compressFile, decompressDir):
        self.compressFile = compressFile
        self.deCompressDir = decompressDir

    def decompressFile(self):
    	supportFormat = ['.7z','.bz2', '.gz', '.rar','.xz', '.zip']
    	fname, fileExt = os.path.splitext(self.compressFile)
        if fileExt == '.tar':
	        tarfile.open(self.compressFile).extractall(self.deCompressDir)
	        return 1
        if fileExt in supportFormat:
 	        Archive(self.compressFile).extractall(self.deCompressDir)
	        return 1
        return 0

    def ParseDecompressDir(self,out_dir,fname):
    	lst = []
        index = 0
        outPath = out_dir + '/' + fname
        for root, dirs, files in os.walk(self.deCompressDir):
            for name in files:
                lst.append(root + '/' + name)

        label = ''
        data = ''

    	for f in lst:
            print f 
            index += 1
            cvparser = CVParser(f,label,out_dir)
            if cvparser.errorMsg:
                print cvparser.errorMsg
                sys.exit(0)
           
            text = cvparser.preprocess(index)
            if text == 0:
                continue
            text = cleanData(text)
            
            data += "\n\n" + text


        print outPath
        if os.path.isfile(outPath):

            opt = raw_input('File exist ... ! Overwrite (Y/N)?\n')
            if ('y' == opt.lower()) :
                fw = open(outPath, "w")
                fw.write(data)
                fw.close()
                print 'overwritten !!...'
            else :
                print "file skipped...."
                return

        fw = open(outPath, "w")
        fw.write(data)
        fw.close()

#### function to uncompress the compressed files
       
    def genSingleFile(self, singleDataFile):
        outDir = self.deCompressDir
        singleDir = outDir + '/singleData'
        if not os.path.exists(singleDir):
            os.mkdir(singleDir)
        self.decompressFile()
        self.ParseDecompressDir(singleDir,singleDataFile)


#### function that returns an ITERATOR that convert a file iteratively

def convertFiles2TextIter(in_dir,label):
    index = 0

    out_dir = ''
    for file1 in os.listdir(in_dir):

        filePath = os.path.join(in_dir, file1)

        if os.path.isfile(filePath):
            index += 1

            cvfile = filePath         
             
            cvparser = CVParser(cvfile,label,out_dir)
            if cvparser.errorMsg:
                print cvparser.errorMsg
                sys.exit(0)
           
            text = cvparser.preprocess(index)
        
            text = cleanse_data(text)
            md5_str = getMD5HashDigest(text)
        
            metaStr = getMetaString(md5_str,label)
    
            data = []
            data.append(file1)
            data.append(metaStr)
            data.append(text)

            yield data 

#### function that convert all files in in_dir and stored all files in out_dir

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
              #  print text
                md5_str = getMD5HashDigest(text)
                #print md5_str
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


### generate label from filename
def genLabel(strg):
    return strg.split("/")[-1]
    
### wrapper for convert files in input_directory and stores in output_directory

def convertDirFiles(dirIn,dirOut):
    label = genLabel(dirIn)
    print label
    convertFiles(dirIn,label,dirOut)

#### generate meta string  from MD5 checksum and label   

def getMetaString(md5Str,label):
    strg = str(md5Str) + "_" + label
    return strg

##### wrapper for convert files in input_directory one by one

def convertFiles2TextIterWrap(dirIn):
    label = genLabel(dirIn)
#    print label
    data1 = convertFiles2TextIter(dirIn,label)
    return data1



if __name__ == '__main__':

    # cvfile = "/home/viswanath/Documents/Blog_ Encoding and Python_ The UnicodeDecodeError exception.html" 
             
    # cvparser = CVParser(cvfile,'','')
    # if cvparser.errorMsg:
    #     print cvparser.errorMsg
    #     sys.exit(0)
           
    # text = cvparser.preprocess(0)
        
    # text = cleanse_data(text)
    # print text

    predict_dir = "/home/viswanath/Downloads/predict17feb"
    predict_out = "/home/viswanath/workspace/code_garage/conver2txt/in_data/predict"
    convertDirFiles(predict_dir,predict_out)


#    obj = decompressFiles('/home/viswanath/Downloads/compressFIle/3LOQ Profiles.tar','/home/viswanath/Downloads/compressFIle/output')
 #   obj.genSingleFile('resultant.txt')