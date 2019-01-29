"""
parameter set for sextractor
"""
def parameter(ch = True):
    """
    record how to change the parameter file and restore
    """
    file_object_1 = open('/home/xkchen/tool/SExtractor/default.param','r+')
    f1 = file_object_1.readlines()
    # next, select the parameter and change 
    keys = [ i.split()[0] for i in f1 if len(i) != 0]
    values = [ ' '.join(i.split()[1:]) for i in f1 if len(i) != 0]
    # find those parameter need to change by the keys dictionary
    a0 = keys.index('#X_IMAGE')
    aa = keys[a0]
    aa = 'X_IMAGE #'
    keys[a0] = aa
    b0 = keys.index('#Y_IMAGE')
    bb = keys[b0]
    keys[b0] = bb
    
    c0 = keys.index('#X_WORLD')
    cc = keys[c0]
    cc = 'X_WORLD #'
    keys[c0] = cc
    d0 = keys.index('#Y_WORLD')
    dd = keys[d0]
    keys[d0] = dd
    # set parameters those need to run with and change the file "default.param"
    keys[0] = ''+ 'NUMBER #'
    f1[0] = ''+keys[0] + values[0] + '\n'
    test_file = open('/home/xkchen/tool/SExtractor/default.param','w') 
    test_file.writelines(f1) # change the txt with f1, re-write the parameter
    test_file.close()
    return
# SEP set for mask A & B
