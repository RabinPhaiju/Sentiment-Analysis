from django.views import View
from django.shortcuts import redirect, render
from sklearn.linear_model import LogisticRegression
import pickle
import os
import string,re
from bs4 import BeautifulSoup
import emoji,contractions as cn

# Create your views here.
class SentimentView(View):
    def get(self,request,format=None):
        algorithm = {'algorithm':[
            'Linear Regression',
            'Random Forest',
            'Logistic Regression',    
        ]}
        return render(request, 'sentiment.html', algorithm)

    def post(self,request):
        gnb = pickle.load(open(os.getcwd()+'/sentiment/gnb.sav','rb'))
        mnb = pickle.load(open(os.getcwd()+'/sentiment/mnb.sav','rb'))
        bnb = pickle.load(open(os.getcwd()+'/sentiment/bnb.sav','rb'))
        nb_tf = pickle.load(open(os.getcwd()+'/sentiment/g_m_b_vectorizer.sav','rb'))

        linear_svc = pickle.load(open(os.getcwd()+'/sentiment/linear_svc.sav','rb'))
        svc_tf = pickle.load(open(os.getcwd()+'/sentiment/linear_svc_vectorizer.sav','rb'))

        rnd_f = pickle.load(open(os.getcwd()+'/sentiment/rnd_f.sav','rb'))
        rnd_tf = pickle.load(open(os.getcwd()+'/sentiment/rnd_f_vectorizer.sav','rb'))

        knn = pickle.load(open(os.getcwd()+'/sentiment/knn_0.799_20,40,2.sav','rb'))
        knn_tf = pickle.load(open(os.getcwd()+'/sentiment/knn_vectorizer.sav','rb'))

        dec_t = pickle.load(open(os.getcwd()+'/sentiment/dec_tree.sav','rb'))
        dec_tf = pickle.load(open(os.getcwd()+'/sentiment/dec_tree_vectorizer.sav','rb'))
        logreg = pickle.load(open(os.getcwd()+'/sentiment/logreg.sav','rb'))
        log_tf = pickle.load(open(os.getcwd()+'/sentiment/logreg_vectorizer.sav','rb'))
        # type = request.data('type')
        # text = request.data.get('text')
        text = request.POST.get('text')
        clean = request.POST.get('clean')
        cleaned_text = self.clean_text(text,clean)
        if len(cleaned_text) !=0:
            po_ne = lambda x: 'pos' if x==1 else 'neg'
            # classes_gnb = gnb.predict(nb_tf.transform(cleaned_text).toarray())
            classes_mnb = mnb.predict(nb_tf.transform(cleaned_text))
            classes_bnb = bnb.predict(nb_tf.transform(cleaned_text))
            decision_t = dec_t.predict(dec_tf.transform(cleaned_text).toarray())
            knn_c = knn.predict(knn_tf.transform(cleaned_text).toarray())
            logRe = logreg.predict(log_tf.transform(cleaned_text).toarray())
            random_f = rnd_f.predict(rnd_tf.transform(cleaned_text).toarray())
            linear = linear_svc.predict(svc_tf.transform(cleaned_text).toarray())
        
            response = []
            for sentence,LOGREs,DECs,KNNs,RNDf,LINEAR_SVC,MNB,BNB in zip(cleaned_text,logRe,decision_t,knn_c,random_f,linear,classes_mnb,classes_bnb):
                if sentence != '':
                    response.append(
                    {'sentence':sentence,'log_po_ne':po_ne(LOGREs),'dec_po_ne':po_ne(DECs),'knn_po_ne':po_ne(KNNs),'rnd_po_ne':po_ne(RNDf),'linear_po_ne':po_ne(LINEAR_SVC),'mnb_po_ne':po_ne(MNB),'bnb_po_ne':po_ne(BNB)}
                    )
            return render(request, 'sentiment.html', {'response':response,'text':text})
            # (sentence,'pos' if (g+m+b+dec+knnC+LOGREs+randomF+svc_lin)>=4 else 'neg')
            

    def clean_text(self,text_input,clean):
        list_sentence = text_input.split('\n')
        if clean == 'clean':
            list_sentence = [self.removing_stopwords(self.cleaner(sentence)) for sentence in list_sentence if len(sentence)!=0]
            print('clean')
        else:
            list_sentence = [sentence for sentence in list_sentence if len(sentence)!=0]
            print('not clean')

        return list_sentence
    
    def removing_stopwords(self,sentence):
        stop_wr={'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own',\
         'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', \
             'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', \
                 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', \
                     'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',\
                          'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', \
                              'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'} 

        filter_tokens = [word for word in sentence.split(' ') if word.lower() not in stop_wr]
        filtered_text = ' '.join(filter_tokens)

        return filtered_text


    def cleaner(self,text_arg):
        """
            Parameters:
            ______________________________
            text_arg: String argument

            Goal:
            ______________________________
            1. converting all letters to lower or upper case
            2. converting numbers into words or removing numbers
            3. removing punctuations, accent marks and other diacritics
            4. removing white spaces
            5. expanding abbreviations
            6. removing stop words, sparse terms, and particular words
            7. text canonicalization

            Returns:
            ______________________________
            processed_string:String
        """

        # remove html tag
        soup = BeautifulSoup(text_arg, "html.parser")
        processed_string = soup.get_text()
        text_arg = re.sub('\[[^]]*\]', '', processed_string)
        
        # remove URLS
        processed_string = re.sub('http[s]?://\S+', '', processed_string)

        # remove hashtags
        processed_string = re.sub('#+',"",processed_string)

        # convert emoji to text
        processed_string = emoji.demojize(processed_string)

        # remove contractions . eg: I've : I have
        processed_string = cn.fix(processed_string)

        # remove special characters
        processed_string = re.sub(r'\W',' ',processed_string)

        # to lower
        processed_string = processed_string.lower()

        # remove all single characters
        processed_string = re.sub(r'\s+[a-zA-Z]\s+',' ',processed_string)

        # remove letter repetation
        processed_string = re.sub(r'(.)\1+',r'\1\1',processed_string)

        # remove punctuation
        processed_string = ''.join(ch for ch in processed_string if ch not in string.punctuation)

        #replace username
        replace_username_with = "user"
        processed_string = re.sub('B\@\w+',replace_username_with,processed_string)

        # remove multiple white space
        processed_string = re.sub('(\s+)',' ',processed_string)

        return processed_string