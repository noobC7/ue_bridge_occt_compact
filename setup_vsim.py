# Import this module to automatically setup path to local ivs module
# This module first tries to see if ivs module is installed via pip
# If it does then we don't do anything else
# Else we look up grand-parent folder to see if it has ivs folder
#    and if it does then we add that in sys.path

import os,sys,inspect,logging

#this class simply tries to see if ivs 
class SetupPath:
    @staticmethod
    def getDirLevels(path):
        path_norm = os.path.normpath(path)
        return len(path_norm.split(os.sep))

    @staticmethod
    def getCurrentPath():
        cur_filepath = os.path.abspath(inspect.getfile(inspect.currentframe()))
        return os.path.dirname(cur_filepath)

    @staticmethod
    def getGrandParentDir():
        cur_path = SetupPath.getCurrentPath()
        if SetupPath.getDirLevels(cur_path) >= 2:
            return os.path.dirname(os.path.dirname(cur_path))
        return ''
    @staticmethod
    def getParentDir():
        cur_path = SetupPath.getCurrentPath()
        if SetupPath.getDirLevels(cur_path) >= 1:
            return os.path.dirname(cur_path)
        return ''
    
    @staticmethod
    def addVSimModulePath():
        # if ivs module is installed then don't do anything else
        #import pkgutil
        #ivs_loader = pkgutil.find_loader('ivs')
        #if ivs_loader is not None:
        #    return

        parent = SetupPath.getGrandParentDir()
        if parent !=  '':
            ivs_path = os.path.join(parent, 'ivs')
            client_path = os.path.join(ivs_path, 'client.py')
            if os.path.exists(client_path):
                sys.path.insert(0, parent)
        else:
            logging.warning("ivs module not found in grandparent folder.")

SetupPath.addVSimModulePath()
