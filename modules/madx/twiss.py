import pandas as pd 



class twiss:
	'''A class to read twiss files and load them into a python object.'''
    def __init__(self,filename):
        self._get_global_quantities(filename)
        self._read_twiss_table(filename)
    
    def _get_global_quantities(self,filename):
        '''Extract global quantities from a twiss file'''
        global_quantities = []
        with open(filename) as f:
            for line in f:
                if "@" in line:
                    name  = line.split()[1]
                    value = line.split()[-1]
                    try:
                        setattr(self, name, float(value))
                    except ValueError:
                        setattr(self, name, value)                        
                    global_quantities.append(name)
                        
                if '$' in line:
                    break            
        self.global_quantities = global_quantities


    def _read_twiss_table(self,filename):    
        with open(filename) as f:
            for nline, line in enumerate(f):
                if '*' in line:
                    line = line.replace("*","")
                    cols = line.split()
                if "$" in line:
                    break
        nline = nline + 1
        twiss = pd.read_csv(filename, skiprows=nline, delim_whitespace=True, names=cols)
        self.table = twiss        
        
