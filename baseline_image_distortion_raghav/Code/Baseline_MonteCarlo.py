# -*- coding: utf-8 -*-
"""
Created on Sun May  5 01:27:10 2019

@author: ramacr1
"""

##
## name: Baseline.py
## purpose: Baseline implementation with Monte Carlo Simulation
## date: 05/01/2019
##
##

import Sandbox
import DLDataTank
from copy import deepcopy
import numpy as np

from timeit import default_timer as timer
# START MY TIMER
start = timer()

if __name__ == '__main__':

    baseline = Sandbox.Sandbox()
    csv = "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DL_info_liver_32_32.csv"
    files = [
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\002648_01_01_061.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\002648_02_01_311.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\002677_01_01_044.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\002680_04_01_050.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\002680_05_01_305.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\002680_06_01_313.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\002680_06_01_323.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\002680_06_01_344.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003599_01_01_247.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003599_01_01_260.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003599_01_01_270.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003620_01_02_280.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003657_01_01_049.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003657_01_01_050.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003657_01_01_069.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003663_01_03_190.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003663_01_03_209.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003663_01_03_237.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003663_01_03_304.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003663_01_03_308.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003663_02_02_276.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\003667_01_01_052.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\004416_01_01_047.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000016_02_01\\253.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000047_10_01\\051.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000047_10_01\\054.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000053_03_01\\044.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000053_04_01\\042.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000053_05_01\\044.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000053_06_01\\222.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000053_06_01\\297.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000069_02_01\\058.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000069_03_01\\058.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000078_01_01\\124.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000078_02_01\\063.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000088_02_01\\306.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000120_02_01\\061.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000122_01_01\\063.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000122_02_01\\062.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000126_02_01\\044.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000126_02_01\\047.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000268_03_01\\058.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000286_03_01\\265.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000286_04_01\\052.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000286_05_01\\049.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000286_05_01\\052.png",
        "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Data\\Images_png\\000286_05_01\\116.png"
    ]
    links = [
        'https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip',
        'https://nihcc.box.com/shared/static/l9e1ys5e48qq8s409ua3uv6uwuko0y5c.zip',
        'https://nihcc.box.com/shared/static/48jotosvbrw0rlke4u88tzadmabcp72r.zip',
        'https://nihcc.box.com/shared/static/xa3rjr6nzej6yfgzj9z6hf97ljpq1wkm.zip',
        'https://nihcc.box.com/shared/static/58ix4lxaadjxvjzq4am5ehpzhdvzl7os.zip',
        'https://nihcc.box.com/shared/static/cfouy1al16n0linxqt504n3macomhdj8.zip',
        'https://nihcc.box.com/shared/static/z84jjstqfrhhlr7jikwsvcdutl7jnk78.zip',
        'https://nihcc.box.com/shared/static/6viu9bqirhjjz34xhd1nttcqurez8654.zip',
        'https://nihcc.box.com/shared/static/9ii2xb6z7869khz9xxrwcx1393a05610.zip',
        'https://nihcc.box.com/shared/static/2c7y53eees3a3vdls5preayjaf0mc3bn.zip',

        'https://nihcc.box.com/shared/static/2zsqpzru46wsp0f99eaag5yiad42iezz.zip',
        'https://nihcc.box.com/shared/static/8v8kfhgyngceiu6cr4sq1o8yftu8162m.zip',
        'https://nihcc.box.com/shared/static/jl8ic5cq84e1ijy6z8h52mhnzfqj36q6.zip',
        'https://nihcc.box.com/shared/static/un990ghdh14hp0k7zm8m4qkqrbc0qfu5.zip',
        'https://nihcc.box.com/shared/static/kxvbvri827o1ssl7l4ji1fngfe0pbt4p.zip',
        'https://nihcc.box.com/shared/static/h1jhw1bee3c08pgk537j02q6ue2brxmb.zip',
        'https://nihcc.box.com/shared/static/78hamrdfzjzevrxqfr95h1jqzdqndi19.zip',
        'https://nihcc.box.com/shared/static/kca6qlkgejyxtsgjgvyoku3z745wbgkc.zip',
        'https://nihcc.box.com/shared/static/e8yrtq31g0d8yhjrl6kjplffbsxoc5aw.zip',
        'https://nihcc.box.com/shared/static/vomu8feie1qembrsfy2yaq36cimvymj8.zip',

        'https://nihcc.box.com/shared/static/ecwyyx47p2jd621wt5c5tc92dselz9nx.zip',
        'https://nihcc.box.com/shared/static/fbnafa8rj00y0b5tq05wld0vbgvxnbpe.zip',
        'https://nihcc.box.com/shared/static/50v75duviqrhaj1h7a1v3gm6iv9d58en.zip',
        'https://nihcc.box.com/shared/static/oylbi4bmcnr2o65id2v9rfnqp16l3hp0.zip',
        'https://nihcc.box.com/shared/static/mw15sn09vriv3f1lrlnh3plz7pxt4hoo.zip',
        'https://nihcc.box.com/shared/static/zi68hd5o6dajgimnw5fiu7sh63kah5sd.zip',
        'https://nihcc.box.com/shared/static/3yiszde3vlklv4xoj1m7k0syqo3yy5ec.zip',
        'https://nihcc.box.com/shared/static/w2v86eshepbix9u3813m70d8zqe735xq.zip',
        'https://nihcc.box.com/shared/static/0cf5w11yvecfq34sd09qol5atzk1a4ql.zip',
        'https://nihcc.box.com/shared/static/275en88yybbvzf7hhsbl6d7kghfxfshi.zip',

        'https://nihcc.box.com/shared/static/l52tpmmkgjlfa065ow8czhivhu5vx27n.zip',
        'https://nihcc.box.com/shared/static/p89awvi7nj0yov1l2o9hzi5l3q183lqe.zip',
        'https://nihcc.box.com/shared/static/or9m7tqbrayvtuppsm4epwsl9rog94o8.zip',
        'https://nihcc.box.com/shared/static/vuac680472w3r7i859b0ng7fcxf71wev.zip',
        'https://nihcc.box.com/shared/static/pllix2czjvoykgbd8syzq9gq5wkofps6.zip',
        'https://nihcc.box.com/shared/static/2dn2kipkkya5zuusll4jlyil3cqzboyk.zip',
        'https://nihcc.box.com/shared/static/peva7rpx9lww6zgpd0n8olpo3b2n05ft.zip',
        'https://nihcc.box.com/shared/static/2fda8akx3r3mhkts4v6mg3si7dipr7rg.zip',
        'https://nihcc.box.com/shared/static/ijd3kwljgpgynfwj0vhj5j5aurzjpwxp.zip',
        'https://nihcc.box.com/shared/static/nc6rwjixplkc5cx983mng9mwe99j8oa2.zip',

        'https://nihcc.box.com/shared/static/rhnfkwctdcb6y92gn7u98pept6qjfaud.zip',
        'https://nihcc.box.com/shared/static/7315e79xqm72osa4869oqkb2o0wayz6k.zip',
        'https://nihcc.box.com/shared/static/4nbwf4j9ejhm2ozv8mz3x9jcji6knhhk.zip',
        'https://nihcc.box.com/shared/static/1lhhx2uc7w14bt70de0bzcja199k62vn.zip',
        'https://nihcc.box.com/shared/static/guho09wmfnlpmg64npz78m4jg5oxqnbo.zip',
        'https://nihcc.box.com/shared/static/epu016ga5dh01s9ynlbioyjbi2dua02x.zip',
        'https://nihcc.box.com/shared/static/b4ebv95vpr55jqghf6bthg92vktocdkg.zip',
        'https://nihcc.box.com/shared/static/byl9pk2y727wpvk0pju4ls4oomz9du6t.zip',
        'https://nihcc.box.com/shared/static/kisfbpualo24dhby243nuyfr8bszkqg1.zip',
        'https://nihcc.box.com/shared/static/rs1s5ouk4l3icu1n6vyf63r2uhmnv6wz.zip',

        'https://nihcc.box.com/shared/static/7tvrneuqt4eq4q1d7lj0fnafn15hu9oj.zip',
        'https://nihcc.box.com/shared/static/gjo530t0dgeci3hizcfdvubr2n3mzmtu.zip',
        'https://nihcc.box.com/shared/static/7x4pvrdu0lhazj83sdee7nr0zj0s1t0v.zip',
        'https://nihcc.box.com/shared/static/z7s2zzdtxe696rlo16cqf5pxahpl8dup.zip',
        'https://nihcc.box.com/shared/static/shr998yp51gf2y5jj7jqxz2ht8lcbril.zip',
        'https://nihcc.box.com/shared/static/kqg4peb9j53ljhrxe3l3zrj4ac6xogif.zip'
    ]
    outputDir = "C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Output"
    length = 32
    width = 32
    result_error = np.array([])
    result_accuracy = np.array([])
    result_sensitivity = np.array([])
    result_specificity = np.array([])
    result_FP = np.array([])
    result_F1score = np.array([])
    result_PPV = np.array([])
    result_NPV = np.array([])
    
    N=10
    #Monte Carlo Simulation, run this a number of times
    while N>=0:

        ## Initialize DL data tank
        baseline.initDLDataTank(length, width) 
                                #(csv,files,links,outputDir,
                                #length, width)

        ## Initialize FFBP
        hiddenLayers = [7, 7]
        eta = 1.1
        baseline.initFFBP(hiddenLayers,eta)

        ## Run Train-Validation-Testing scheme
        trainingLen = 22
        validationLen = 10
        testingLen = 10
        maxEpochs = 10
        errorThreshold = 0.05
        classificationThreshold = 0.5
        baseline.setLesionThreshold(classificationThreshold)
        baseline.runExperiment(trainingLen,
                            validationLen,
                            testingLen,
                            maxEpochs,
                            errorThreshold, orig_ds, num_images) #"gaussian",0) 
                            #rotate180_orig_ds, num_images) #mixed_ds, Gnoise_orig_ds, Gamma_orig_ds

        confusionMatrix = baseline.getConfusionMatrix()
        TN =  confusionMatrix[0,0]
        FP = confusionMatrix[0,1]
        FN = confusionMatrix[1,0]
        TP = confusionMatrix[1,1]
        result_error = np.append(result_error, [(FP+FN) /(TN+FP+FN+TP)],axis=0)
        #print("error " + str((TP+TN) /(TN+FP+FN+TP)))

        result_accuracy = np.append(result_accuracy,(TP+TN)/(TN+FP+FN+TP))
        #print("accuracy " + str((FP+FN)/(TN+FP+FN+TP)))

        result_sensitivity = np.append(result_sensitivity,TP/(TP+FN))
        #print("sensitivity " + str(TP/(TP+FN)))

        result_specificity =np.append(result_specificity,TN/(TN+FP))
        # print("specificity " + str(TN/(TN+FP)))

        result_FP = np.append(result_FP, FP)
        #print("false positives " + str(FP))
        
        result_PPV = np.append(result_PPV, TP/(TP+FP))

        result_NPV = np.append(result_NPV, TN/(TN+FN))

        result_F1score = np.append(result_F1score, 2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP)) + (TP/(TP+FN))))
        
        N -=1
        print(N)

    # STOP MY TIMER
    elapsed_time = timer() - start  # in seconds
    print('elapsed time '+ str(elapsed_time))
    #print(result_error)
    print('average error ' + str(np.nansum(result_error) / (len(result_error) - sum(np.isnan(result_error)))))
    #print(result_accuracy)
    print('average accuracy ' + str(np.nansum(result_accuracy) / (len(result_accuracy) - sum(np.isnan(result_accuracy)))))
    #print(result_sensitivity)
    print('average sensitivity ' + str(np.nansum(result_sensitivity) / (len(result_sensitivity) - sum(np.isnan(result_sensitivity)))))
    # print(result_specificity)
    print('average specificity ' + str(np.nansum(result_specificity) / (len(result_specificity) - sum(np.isnan(result_specificity)))))
    #print(result_FP)
    print('average FP ' + str(np.nansum(result_FP) / (len(result_FP) - sum(np.isnan(result_FP)))))
    print('average PPV ' + str(np.nansum(result_PPV) / (len(result_PPV) - sum(np.isnan(result_PPV)))))
    print('average NPV ' + str(np.nansum(result_NPV) / (len(result_NPV) - sum(np.isnan(result_NPV)))))
    print('average F1-score ' + str(np.nansum(result_F1score) / (len(result_F1score) - sum(np.isnan(result_F1score)))))