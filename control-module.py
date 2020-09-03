
#read in the emotion variable predicted from the AI
f = open( "emo.txt", O_RDONLY|O_CREAT )
pred_f_emo = read(f, 50)
print(pred_f_emo)

#display set the corrisponding AUs based on each emotion
if (pred_f_emo == 'Happy'):
    bml.execBML('*', '<face amount="1" au="6" end="1" side="BOTH" start="0" type="facs"/>') 
    bml.execBML('*', '<face amount="1" au="12" end="1" side="BOTH" start="0" type="facs"/>')
elif(pred_f_emo == 'Angery'):
    bml.execBML('*', '<face amount="1" au="4" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="5" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="7" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="10" end="1" side="BOTH" start="0" type="facs"/>')
elif(pred_f_emo == 'Disgust'):
    bml.execBML('*', '<face amount="1" au="6" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="7" end="0.5" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="10" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="25" end="1" side="BOTH" start="0" type="facs"/>')
elif(pred_f_emo == 'Fearful'):
    bml.execBML('*', '<face amount="1" au="1" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="2" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="4" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="5" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="7" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="10" end="1" side="BOTH" start="0" type="facs"/>')
elif(pred_f_emo == 'Sad'):
    bml.execBML('*', '<face amount="1" au="1" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="4" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="10" end="1" side="BOTH" start="0" type="facs"/>')
elif(pred_f_emo == 'Surprise'):
    bml.execBML('*', '<face amount="1" au="1" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="2" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="5" end="1" side="BOTH" start="0" type="facs"/>')
    bml.execBML('*', '<face amount="1" au="26" end="1" side="BOTH" start="0" type="facs"/>')
