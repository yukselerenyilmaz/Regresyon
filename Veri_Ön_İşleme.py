#!/usr/bin/env python
# coding: utf-8

# # Kayıp / Eksik Verilerin Giderilmesi (Imputation)
# 
# Çeşitli nedenlerle birçok gerçek dünya veri kümesi, genellikle boşluklar, NaN veriler gibi eksik değerler içerir. Ancak bu tür veri kümeleri, bir dizideki tüm değerlerin sayısal olduğunu ve hepsinin anlam taşıdığını varsayan scikit-learn kütüphanesi tahmin algoritmaları ile uyumlu değildir. Eksik veri kümelerini kullanmak için temel bir strateji, eksik değerleri içeren tüm satırları ve / veya sütunları atmaktır. Ancak bu işlem değerli olabilecek (eksik olsa bile) veriyi kaybetme anlamına gelir. Daha iyi bir strateji, eksik verilerin değerlerini belirlemek, yani bunları verilerin bilinen kısmından çıkarmaktır.
# 
# Scikit-learn kütüphanesinde `SimpleImputer` sınıfı, eksik değerleri hesaplamak için temel stratejiler sağlar. Eksik değerler, sağlanan sabit bir değer ile veya eksik değerlerin bulunduğu her bir sütunun istatistikleri (ortalama, medyan veya en sık) kullanılarak hesaplanabilir. 
# 
# 

# In[1]:


import pandas as pd
import numpy as np


df = pd.DataFrame([["a", "x"],
                    [np.nan, "y"],
                    ["a", np.nan],
                    ["b", "y"]], dtype="category")


from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent") 
print(imp.fit_transform(df))


# In[3]:


import pandas as pd
veriseti = pd.read_csv("housing.csv")
veriseti.head()


# In[4]:


veriseti.isnull().values.any()


# In[5]:


veriseti.isnull().sum().sum()


# In[6]:


veriseti['total_bedrooms'].isnull().sum()


# In[7]:


# Bagimsiz Degiskenler
X = veriseti.iloc[:,2:8].values  
# median_house_value bagimli degisken (tahmin edilecek olan) olsun
y = veriseti.iloc[:,-2].values # Bagimli Degisken


# In[8]:


from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean') # Strategy parametresi olarak mean ifadesini kullanıyoruz.
impute = imputer.fit(X[:,2:8])  # fit ile ögrenme
X[:,2:8]=imputer.transform(X[:,2:8])  # transform ile ögrenileni uygulama
print(X)


# In[9]:


print(np.isnan(np.sum(X)))


# `KNNImputer` sınıfı, k-en yakın komşu yaklaşımını kullanarak eksik değerleri doldurmayı sağlar. Varsayılan olarak, en yakın komşuları bulmak için bir öklid mesafe ölçüsü kullanılır. Her eksik özellik, o özellik için bir değere sahip en yakın komşuların değerleri kullanılarak belirlenir. Komşuların özelliklerinin ortalaması tekdüze olarak hesaplanır veya her bir komşuya olan mesafeye göre ağırlıklandırılır.

# In[10]:


from sklearn.impute import KNNImputer
# Bagimsiz Degiskenler
X_knn = veriseti.iloc[:,2:8].values  
# median_house_value bagimli degisken (tahmin edilecek olan) olsun
y = veriseti.iloc[:,-2].values # Bagimli Degisken
imputer = KNNImputer(n_neighbors=2, weights="uniform")
imputer.fit_transform(X_knn)


# In[11]:


(X==X_knn).all()


# # Kategorik Verilerin Dönüşümü
# 
# Birçok makine öğrenmesi çalışmasında, veri kümesi metin veya kategorik değerler (temelde sayısal olmayan değerler) içerebilir. Örneğin kırmızı, turuncu, mavi, beyaz vb. eğerlere sahip renk özelliği ya da kahvaltı, öğle yemeği, ara öğünler, akşam yemeği, çay vb. değerlere sahip yemek planı verilebilir.
# 
# Yapay zeka ve makine öğrenmesi için, algoritmaların çoğunun sayısal girdilerle daha iyi çalıştığını fark edeceksiniz. Bu nedenle, bir analistin karşılaştığı ana zorluk, metin / kategorik verileri sayısal verilere dönüştürmek ve yine de bundan anlam çıkarmak için bir algoritma / model oluşturmaktır.
# 
# Kategorik değerleri sayısal değerlere dönüştürmenin birçok yolu vardır. Her yaklaşımın kendi ödünleşimleri ve özellik seti üzerinde etkisi vardır. Bu bağlamda, iki ana yönteme odaklanacağız: One-Hot-Encoding ve Label-Encoder. 
# 
# Bu dönüşüm yöntemlerinin her ikisi de SciKit-learn kütüphanesinin parçasıdır ve metin veya kategorik verileri modelin beklediği ve daha iyi performans göstereceği sayısal verilere dönüştürmek için kullanılır.
# 
# ## Label Encoding

# In[12]:


import pandas as pd
veriseti = pd.read_csv("housing.csv")
veriseti.head()


# In[13]:


veriseti['ocean_proximity'].unique()


# In[14]:


# Bagimsiz Degiskenler
X = veriseti.iloc[:,2:10].values  
# median_house_value bagimli degisken (tahmin edilecek olan) olsun
y = veriseti.iloc[:,-2].values # Bagimli Degisken


# In[15]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
print(X)


# In[16]:


np.unique(X[:,-1])


# Numaralandırma sıralı seri halinde (yukarıdan aşağıya) değil, kategori isminin alfabetik sırasına göre yapılmaktadır. 

# ## One-Hot Encoding
# 

# 3 yiyecek kategoriniz olduğunu düşünün: elma, tavuk ve brokoli. Etiket kodlamasını (Label-encoding) kullanarak, bunları kategorize etmek için bunların her birine bir sayı atarsınız: elma = 1, tavuk = 2 ve brokoli = 3. Ancak şimdi, modelinizin kategoriler arasında ortalamayı dahili olarak hesaplaması gerekiyorsa, (1+3) = 4/2 = 2. Bu, modelinize göre elma ve tavuğun birlikte ortalamasının brokoli olduğu anlamına gelir.
# 
# Modelinizin bu düşünce şekli, ilişkilerin tamamen yanlış kurulmasına yol açacak, bu yüzden one-hot encoding kavramını sunmamız gerekiyor.
# 
# Her şeyi 1'den başlayan ve ardından her kategori için artan bir sayı olarak etiketlemek yerine, daha çok ikili kategorilere ayırmaya gideceğiz. 
# 
# >![One-Hot Encoding](https://miro.medium.com/max/2736/0*T5jaa2othYfXZX9W.)
# 
# Fark nedir? Bizim kategorilerimiz eskiden satırlardı ama şimdi sütun oldular. Sayısal değişken olan kaloriler yine aynı kaldı. Belirli bir sütundaki 1 değeri, bilgisayara o satırın verileri için doğru kategoriyi söyleyecektir. Diğer bir deyişle, her kategori için ek bir ikili sütun oluşturduk.
# 
# Şimdi housing.csv veri seti üzerinde deneyelim.

# In[17]:


import pandas as pd
veriseti = pd.read_csv("housing.csv")
veriseti.head()


# In[18]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Bagimsiz Degiskenler
X = veriseti.iloc[:,2:10].values  
# median_house_value bagimli degisken (tahmin edilecek olan) olsun
y = veriseti.iloc[:,-2].values # Bagimli Degisken

labelencoder_X=LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
print(X)


# In[19]:


okyanus_yakinligi = veriseti.iloc[:,-1].values

onehotencoder_X = OneHotEncoder()  # OneHotEncoder = Atanan sayısal değerlerin 0-1 olarak ulke değerlerine işlenmesi
X = onehotencoder_X.fit_transform(okyanus_yakinligi.reshape(-1, 1)).toarray() #Dizinin içine yerleştirme işlemi
print(X)


# # Standardizasyon - Ölçeklendirme (Scaler)
# Bir veri kümesinin standardizasyonu, birçok makine öğrenmesi tahmin algoritması için ortak bir gerekliliktir. Verilerin kendine özgü özellikleri aşağı yukarı standart olarak dağıtılmış veriler gibi görünmüyorsa kötü davranabilirler.
# 
# Örneğin, bir öğrenme algoritması tüm verilerin ortalandığını ve benzer varyansa sahip olduğunu varsayar. Bir özelliğin diğerlerinden daha büyük olan bir varyansı varsa, tahmin algoritmasının doğru bir şekilde öğrenmesini engelleyebilir.
# 

# In[21]:


from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])


# In[22]:


scaler = preprocessing.StandardScaler().fit(X_train)


# In[23]:


X_scaled = scaler.transform(X_train)
X_scaled


# In[24]:


X_scaled.mean(axis=0)


# In[25]:


X_scaled.std(axis=0)


# Alternatif bir standardizasyon , belirli bir minimum ve maksimum değer arasında ölçeklendirmedir.
# 
# Bir oyuncak matrisi [0, 1] aralığında ölçeklendirelim.

# In[26]:


X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax


# In[27]:


min_max_scaler2 = preprocessing.MinMaxScaler(feature_range=(0.2, 0.8))


# In[28]:


X_train_minmax2 = min_max_scaler2.fit_transform(X_train)
X_train_minmax2

