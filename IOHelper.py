import os
import re
import shutil

class IOHelper:

    def __init__(self,data_dir):
        #TODO make more generic
        self.dir = os.getcwd()
        self.input_dir = data_dir
        self.outdir = os.path.join(data_dir,"output")
        self.write_mode = 'a+'
        self.filename = 'network_summary.csv'
        self.out_file_path = os.path.join(self.outdir,self.filename)
        #TODO: Extend to reach all formats supported by networkx
        self.supported_formats = ['gz','bz2','adjlist','GXL','GML','XGMML','SVG','GEXF','net','txt']
        self.ignore_list = []
        self.load_ignore_list()
        self.locate_additional_ignores()

    def ends_with_supported_format(self,file):
        r = False
        for x in self.supported_formats:
            if(file.endswith(x)):
                r = True
                break
        return r

    #Boolean function to check if file is valid network file and return false otherwise.
    def is_network_file(self,file):
        #TODO: figure out why python is adding extra backslash and breaking regex
        #combined_regex = r'(\.' + r')|(\.'.join(x for x in self.supported_formats) + r')'
        # if ((re.match(combined_regex,file))):
        #     return True
        if(self.ends_with_supported_format(file) and not file.startswith('.') and file not in self.ignore_list):
            return True
        return False

    def find_ignores_from_output(self):
        scanpath = os.path.join(self.outdir,self.filename)
        if(not os.path.isfile(scanpath)):
            return []
        linecount = 0
        ignore_list = []
        scanfile = open(scanpath,'r')
        lines = scanfile.readlines()
        for line in lines:
            if line == '':
                continue
            if linecount == 0:
                linecount+=1
                continue
            cleanline = line.strip()
            line_parts = line.split(',')
            path_split = os.path.split(line_parts[0])
            ignore_list.append(path_split[-1])
        return ignore_list


    # Get ignore networks from ignore file if it exists
    #TODO: LOAD FROM OUTPUT FILE
    def load_ignore_list(self):
        self.ignore_list = self.find_ignores_from_output()
        if len(self.ignore_list)==0:
            print('no previous output found... starting fresh')
        else:
            print('ignoring {}'.format(','.join(x for x in self.ignore_list)))
        return

    def locate_additional_ignores(self):
        tracker = []
        searchpath = self.getCurrentDirectory()
        ignore_path = os.path.join(searchpath,'ignore.txt')
        if(os.path.exists(ignore_path) and os.path.isfile(ignore_path)):
            ignore_file = open(ignore_path,'r')
            ignore_lines = ignore_file.readlines()
            for line in ignore_lines:
                if line.startswith('#'):
                    continue
                if line.strip() not in self.ignore_list:
                    self.ignore_list.append(line.strip())
                    tracker.append(line.strip())
            print('finished loading additional ignores {}'.format(','.join(x for x in tracker)))
            return
        print('no additional ignores found')
        return


    #TODO: modify function to read newtork using correct loading function
    def get_network_from_file(self,filepath):
        print("Reading repo input file: {}".format(filepath))
        repolist = []
        if ( not os.path.exists(filepath)):
            print("File {} doesn't exist".format(filepath))
            exit(1)
        file = open(filepath,'r')
        filelines = file.readlines()
        for line in filelines:
            line = line.strip()
            if line.startswith('http') and line.endswith('.git'):
                if line not in repolist:
                    repolist.append(line)
            else:
                print("{} does not look like a repo...please check for next run".format(line))
        file.close()
        return repolist

    def getCurrentDirectory(self):
        return self.dir

    def get_files_in_dir(self,dir):
        accepted_files = []
        for filename in os.listdir(self.input_dir):
            if self.is_network_file(filename):
                accepted_files.append(os.path.join(self.input_dir,filename))
        return accepted_files

    def createOutputDirectory(self):
        if (not (os.path.exists(self.outdir))):
            print("Creating output file directory")
            os.makedirs(self.outdir)
        return self.outdir

    def write_output_headers(self):
        self.writeOutputHeader()

    #write header if it doesn't exist
    def writeOutputHeader(self,headerstring):
        output_exists = False
        self.createOutputDirectory()
        if(os.path.isfile(self.out_file_path)):
            output_exists = True
        if not output_exists:
            self.outfile = open(self.out_file_path,self.write_mode)
            self.outfile.writelines(headerstring)
            self.outfile.close()
        return

    def write_out_line(self,line):
        self.outfile = open(self.out_file_path,self.write_mode)
        self.outfile.writelines(line)
        self.outfile.close()
        return

def main():
    print("Welcome to IOHelper Main")
    m = IOHelper()

if __name__ == '__main__':
    main()
