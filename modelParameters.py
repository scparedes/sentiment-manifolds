trainingCount = 25000

testingCount=11000

#index for words that our out-of-vocab; using 0 may not be the best option
#because many weights in the model will vanish so try small positive ints
UNK_INDEX = 0

#assuming full vocab is 85000; much higher in reality
VocabSize_w = (73000 +(UNK_INDEX+1)*(UNK_INDEX>0))

#total number unique chars in reviews (no unicode)
VocabSize_c = (58+(UNK_INDEX+1)*(UNK_INDEX>0))

# median review length in words is 2128
MaxLen_w = 540

# 13.5 is median word length of training data
MaxLen_c = (14*MaxLen_w)



#only valid for words
skip_top = 5




#data run history: 101x1012 (word mini), 101x85000 (word full),
#533x58 (first char)
