#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import Tkinter
import tkFileDialog
from os import listdir
from os.path import isfile, join



class simple_ui(Tkinter.Tk):
    def __init__(self,parent):
        Tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.tclCol = 5

        self.initialize()

    def initialize(self):
        self.grid()

        self.HeaderVar = Tkinter.StringVar()
        labelHead = Tkinter.Label(self,textvariable=self.HeaderVar,
                              anchor="w",fg="red")
        labelHead.grid(column=0,row=0,columnspan=2,rowspan=5,sticky='EW')
        self.HeaderVar.set(u"Candidate Shortlisting System !...")

        self.entryVariable = Tkinter.StringVar()
        self.entry = Tkinter.Entry(self,textvariable=self.entryVariable)
        self.entry.grid(column=0,row=1,sticky='EW')
        self.entry.bind("<Return>", self.OnPressEnter)
        self.entryVariable.set(u"Enter text here.")

        button = Tkinter.Button(self,text=u"Click me !",
                                command=self.OnButtonClick)
        button.grid(column=1,row=1)

        scrollbar = Tkinter.Scrollbar(self)
    #    scrollbar.grid( side = 'right', fill = 'y')

        Tkinter.Label(text='Name',fg="purple",bg="green",width=15,pady=10,padx=5).grid(row=4,column=0,columnspan=2,sticky='EW')
        Tkinter.Label(text='Type',fg="cyan",bg="red",width=10,pady=10,padx=5).grid(row=4,column=1,columnspan=2,sticky='EW')


        self.labelVariable = Tkinter.StringVar()
        label = Tkinter.Label(self,textvariable=self.labelVariable,
                              anchor="w",fg="white",bg="blue")
        label.grid(column=0,row=2,columnspan=2,sticky='EW')
        self.labelVariable.set(u"Hello !")

        self.grid_columnconfigure(0,weight=1)
   #     self.resizable(True,True)
        self.update()
  #      self.geometry(self.geometry())
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)


    def OnButtonClick(self):

        root = Tkinter.Tk()
        msg = "Error No File selected by the user :/...."
        dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
        if len(dirname ) > 0:
            msg = dirname
            self.labelVariable.set( msg+" is selected by user" )
            onlyfiles = [f for f in listdir(dirname) if isfile(join(dirname, f))]
            for file_path in onlyfiles:
                msg = file_path
                print file_path

                f = file_path[-10:]
       # print f
                tname = f.split('.')[1]

                if tname == 'py':
                    tname = 'python'
                elif tname == 'txt':
                    tname = 'text'
                elif tname == 'doc':
                    tname = 'Document'
                elif tname == 'tsv':
                    tname = 'tab separated values'
                elif tname == 'csv':
                    tname = 'comma separated values'
                else:
                    continue

                self.genRow(file_path,tname)

        else:
            self.labelVariable.set( msg )
      #  file_path = tkFileDialog.askopenfilename()
        print msg

        self.entryVariable.set(msg)
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

    def OnPressEnter(self,event):

        root = Tkinter.Tk()
        msg = "Error No File selected by the user :/...."
        # dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
        # if len(dirname ) > 0:
        #     msg = dirname
        #     self.labelVariable.set( msg+" is selected by user" )
        # else:
        #     self.labelVariable.set( msg )
        file_path = tkFileDialog.askopenfilename()
        msg = file_path
        print file_path

        f = file_path[-10:]
        print f
        tname = f.split('.')[1]

        if tname == 'py':
            tname = 'python'
        elif tname == 'txt':
            tname = 'text'
        else:
            tname = 'Unknown Format'

        self.genRow(file_path,tname)

        print msg

        self.entryVariable.set(msg)
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)




    def genRow(self, valCol1,valCol2):

        print "hello inserted ...." + str(self.tclCol)

        Tkinter.Label(text=valCol1,fg="green",bg="yellow",width=15,pady=10,padx=5).grid(row=self.tclCol,column=0,columnspan=2,sticky='EW')
        Tkinter.Label(text=valCol2,fg="orange",bg="blue",width=10,pady=10,padx=5).grid(row=self.tclCol,column=1,columnspan=2,sticky='EW')

        self.tclCol+=1

    def genRowDict(self,dict):
        for name in dict:
            print "hello inserted ...." + str(self.tclCol)

            Tkinter.Label(text=name,fg="green",bg="yellow",width=15,pady=10,padx=5).grid(row=self.tclCol,column=0,columnspan=2,sticky='EW')
            Tkinter.Label(text=dict[name],fg="orange",bg="blue",width=10,pady=10,padx=5).grid(row=self.tclCol,column=1,columnspan=2,sticky='EW')

            self.tclCol+=1



if __name__ == "__main__":

    dict = {}
    dict["sagar"] = 'human'
    dict['brilliant'] = 'tutorials'
    dict['boost'] = 'drink'
    dict['batmanVSsuperman'] = 'movie'

    app = simple_ui(None)
    app.title('my application')

    app.genRow("Pokemon","Cartoon")
    app.genRow("India Today","Magazine")
    app.genRow("NIT","College")
    app.genRow("3loq","company")
    app.genRow("Hockey","Sports")

    app.genRowDict(dict)

    app.mainloop()