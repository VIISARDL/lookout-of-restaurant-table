from __future__ import print_function

import time
import os
from pytorchYolo2.cfg import parse_cfg


class ConfigProvider():
    columns = ['train', 'valid', 'names', 'Backup']
    
    imgFile = 'imgList.txt'
    trainFile = 'train_list.txt'
    testFile = 'test_list.txt'
    
    pathSyn = '~/.Datasets'
    cwd = os.getcwd()
    configRoot = 'Config/'
    projectRoot = cwd.replace('lib', '')
    pathNames = ''
    
    trainList = []
    testList = []
    trainSample = 0
    testSample = 0
    num_workers = 3
    
    seed          = int(time.time())
    eps           = 1e-5
    save_interval = 5  # epoches
    dot_interval  = 70 # batches
    conf_thresh   = 0.25
    nms_thresh    = 0.4
    iou_thresh    = 0.5
    
    USE_CUDA = True # The actual PytorchYolo2 code just work in CUDA
    
    def __init__(self, datacfg='', cfgfile='', weightfile=None):
        self.datacfg = datacfg
        self.cfgfile = cfgfile

        self.__readDataCfg()
        self.__readCfg()
        self.__CreateFullList()
        
    def getTestUrl(self):
        return (self.projectRoot + '/' + self.configRoot + '/' + self.testFile)
    
    def getTrainUrl(self):
        return (self.projectRoot + '/' + self.configRoot + '/' + self.trainFile)
    
    def __readCfg(self):
        if self.cfgfile is not '':
            self.options = parse_cfg(self.cfgfile)[0]
            
            self.batch_size    = int(self.options['batch'])
            self.max_batches   = int(self.options['max_batches'])
            self.learning_rate = float(self.options['learning_rate'])
            self.momentum      = float(self.options['momentum'])
            self.decay         = float(self.options['decay'])
            self.steps         = [float(step) for step in self.options['steps'].split(',')]
            self.scales        = [float(scale) for scale in self.options['scales'].split(',')]

            self.num_epochs = 5 #int(n_inter / (len(train_dataset)/ batch_sz))
    
    def __readDataCfg(self):
        if self.datacfg is not '':
            
            with open(self.datacfg, 'r') as fp:
                lines = fp.readlines()
                
            trainUrl = lines[0].replace('\n','').split(',')[1:]
            for i in range(len(trainUrl)):
                self.trainList.append(self.pathSyn 
                                      + '/' + trainUrl[i] + '/' + self.imgFile)
                
            
            testUrl = lines[1].replace('\n','').split(',')[1:]
            for i in range(len(testUrl)):
                self.testList.append(self.pathSyn 
                                     + '/' + testUrl[i] + '/' + self.imgFile)
                
            nameUrl = lines[2].replace('\n','').split(',')[1:]
            backupUrl = lines[3].replace('\n','').split(',')[1:]
            
            self.backupDir = backupUrl if len(backupUrl) > 0 else 'Backup'
            if not os.path.exists(self.backupDir):
                os.mkdir(self.backupDir)
    
            self.pathNames = self.cwd + '/' + self.configRoot + nameUrl[0]
            
            
    def __CreateFullList(self):
        self.trainSample = 0
        with open(self.projectRoot +'/'+ self.configRoot+self.trainFile, 'w') as f:     
            for fl in self.trainList:
                with open(os.path.expanduser(fl), 'r') as tl:
                    lines = tl.readlines()
                    for line in lines:
                        f.write(line)
                    self.trainSample = len(lines)
                        
        with open(self.projectRoot +'/' + self.configRoot+self.testFile, 'w') as f:     
            for fl in self.testList:
                with open(os.path.expanduser(fl), 'r') as tl:
                    lines = tl.readlines()
                    for line in lines:
                        f.write(line)
                    self.testSample = len(lines)
            

    def createDataCfg(self):
        """
        debug only, it will be delete in a few updates
        """
        trainList = ['Planet']
        validList = ['Planet']
        names = ['cc.names']
        backup = []
        textList = [trainList, validList, names, backup]
        with open(self.datacfg, 'w') as f:
             for i in range(4):
                 lenLine = len(textList[i])
                 f.write(self.columns[i])
                 f.write(',') if lenLine is not 0 else ''
                 
                 for j in range(lenLine):
                     f.write(textList[i][j])
                     if j is not (lenLine - 1):
                         f.write(',')
                 f.write('\n')
                 
        


if __name__ == "__main__":
    #createDataCfg()
    alfa = ConfigProvider('../Config/templateData.csv', '../Config/yolo-voc.cfg')
    alfa.createDataCfg()
    alfa.testSample
    alfa.trainSample
    

    