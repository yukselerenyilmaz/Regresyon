#!/usr/bin/env python
# coding: utf-8

# Doğrusal Regresyon, genellikle her veri bilimcisinin karşılaştığı ilk makine öğrenimi algoritmasıdır. Basit bir model olmasına karşın, diğer makine öğrenimi algoritmalarının temelini oluşturduğu için herkesin bu modelde uzmanlaşması gerekir.
# 
# ## Doğrusal Regresyon Ne İçin Kullanılır? 
# Doğrusal regresyon çok güçlü bir tekniktir ve kârlılığı etkileyen faktörleri irdelemek için bu teknikten faydanılabilir. Örneğin, önceki aylara ait satış verileri analiz edilerek önümüzdeki aylardaki satışları tahmin etmek için kullanılabilir. Müşteri davranışları hakkında çeşitli içgörüler elde etmek için de kullanılabilir. 
# 
# ## Tanım
# Doğrusal bir regresyon modelinin amacı, bir veya daha fazla özellik (bağımsız değişkenler) ile sürekli bir hedef değişken (bağımlı değişken) arasında bir ilişki bulmaktır. Yalnızca özellik olduğunda, buna Tek Değişkenli Doğrusal Regresyon denir. Birden fazla özellik varsa buna Çoklu Doğrusal Regresyon denir.
# 
# ## Hipotez
# Doğrusal Regresyon, tek bir yordayıcı (öngörücü - bağımsız) değişken $x$ temelinde bir $y$ yanıtını tahmin etmenin bir yoludur. $x$  ve $y$ arasında yaklaşık olarak doğrusal bir ilişki olduğu varsayılır. Matematiksel olarak, bu ilişkiyi şu şekilde gösterebiliriz:
# 
# 
# $\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p$
# 
# Burada $w = (w_1,..., w_p)$, doğrusal modelde katsayıları ve $w_0$ ise kesen noktayı temsil eden iki bilinmeyen sabittir.
# 
# Basit Doğrusal Regresyon, katsayıları olan ($w = (w_1,..., w_p)$) doğrusal bir modele uyar. Veri kümesinde gözlemlenen hedefler ile doğrusal yaklaşımla tahmin edilen hedefler arasındaki kalan kareler toplamını en aza indirmeyi hedefler (En Küçük Kareler Yöntemi - Least Squares Method). 
# 
# Aşağıdaki örnek, iki boyutlu grafik içindeki veri noktalarını göstermek için veri setinin yalnızca ilk özelliğini kullanır. Düz çizgi grafikte görülmektedir. Bu çizgi, doğrusal regresyonun veri kümesinde gözlemlenen değerler ile tahmin edilen yanıtlar arasındaki kalan kareler toplamını en aza indirecek düz bir çizgi çizmeye nasıl çalıştığını gösterir.
# >![Basit Doğrusal Regresyon](https://scikit-learn.org/stable/_images/sphx_glr_plot_ols_001.png)
# 
# ## Örnek Çalışma
# 
# 
# Öncelikle gerekli kütüphaneleri yükleyelim: 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Bir sonraki adım, verileri içe aktarmak ve kontrol etmektir. Kaggle üzerinden ABD'deki konut fiyatları ile ilgili verileri barındıran [USA_Housing.csv](https://www.kaggle.com/aariyan101/usa-housingcsv) dosyasını indirelim.  Veri setini keşfetmek her zaman iyi bir uygulamadır. Veri kümesiyle ilgili tüm olası bilgileri almak için kendi verinizi kullanmayı deneyin ve aşağıdaki kodu çalıştırın.

# In[2]:


from google.colab import files
uploaded = files.upload()


# In[3]:


USAhousing = pd.read_csv('USA_Housing.csv')
USAhousing.head()


# Burada, fiyat (Price) sütununu bağımlı değişken olarak ve geri kalanını bağımsız değişkenler olarak kabul edelim. Bu, bağımsız değişkenler verildiğinde fiyatı tahmin etmemiz gerektiği anlamına gelir.
# 
# Şimdi bazı görselleştirmeler yapalım.

# In[4]:


sns.pairplot(USAhousing)


# Bu çiftli grafikler, histogram ve dağılım (scatter) grafiği olmak üzere iki türdür. Köşegen üzerindeki histogram, tek bir değişkenin dağılımını görmemize izin verirken, üst ve alt üçgenlerdeki dağılım grafikleri iki değişken arasındaki ilişkiyi (veya bunların eksikliğini) gösterir.

# In[5]:


sns.distplot(USAhousing['Price'])


# Tek bir değişkeni keşfetmeye başlamanın harika bir yolu histogramdır. Histogram değişkeni bölmelere ayırır, her bölmedeki veri noktalarını sayar ve x eksenindeki bölmeleri ve y eksenindeki sayıları gösterir.

# ## Korelasyon
# Korelasyon katsayısı veya basitçe korelasyon, -1 ile 1 arasında değişen bir indekstir. Değer sıfıra yakın olduğunda, doğrusal bir ilişki yoktur. Korelasyon artı veya eksi bire yaklaştıkça, ilişki daha güçlüdür. Bir (veya negatif olan) değeri, iki değişken arasında mükemmel bir doğrusal ilişkiyi gösterir.
# Veri kümesindeki değişkenler arasındaki korelasyonu bulalım.
# 

# In[6]:


USAhousing.corr()


# In[7]:


plt.figure(figsize=(6, 6))
heatmap = sns.heatmap(USAhousing.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# 
# ## Doğrusal Regresyon Modelinin Eğitimi
# 
# Şimdi regresyon modelini eğitmeye başlayalım. Öncelikle verilerimizi eğitilecek özellikleri içeren bir X dizisine ve hedef değişkenli bir y dizisine, bu durumda Fiyat sütununa ayırmamız gerekecek. Adres sütununu atacağız çünkü sadece doğrusal regresyon modelinin kullanamayacağı metin bilgisine sahip.

# In[14]:


X = USAhousing['Avg. Area Income'].values
y = USAhousing['Price'].values

print(X.shape)
print(type(X))


# In[15]:


uzunluk = len(X)
X = X.reshape((uzunluk,1))
print(X.shape)
print(type(X))


# ## Eğitim - Test Verisi Ayrımı
# 
# Amacımız, yeni verilere iyi genelleyen bir model oluşturmaktır. Test setimiz yeni veriler için bir temsilci görevi görür. Eğitim verileri ile doğrusal regresyon algoritmasını uygular ve modeli oluştururuz. Modelin başarısını denetleyebilmemiz için test verilerini kullanılır. Veri setini eğitim-test olarak bölme işi aşağıdaki gibidir:
# 

# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# Yukarıdaki kod parçacığından, verilerin% 40'ının test verilerine gittiği ve geri kalanının eğitim setinde kaldığı sonucuna varabiliriz.
# 
# ## Modeli Oluşturma ve Eğitme
# 
# 

# In[17]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# Yukarıdaki kod, eğitim verilerindeki doğrusal regresyon modeline uyar.
# 
# ## Modelimizden Tahminler
# Test setinden tahminleri alalım ve ne kadar başarılı olduğunu görelim!

# In[18]:


predictions = lm.predict(X_test)


# Tahmini görselleştirelim

# In[19]:


sns.scatterplot(y_test,predictions)


# In[20]:


lm.score(X_test, y_test)


# # Örnek Çalışma 2

# In[21]:


from google.colab import files
uploaded = files.upload()


# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bas_beyin = pd.read_csv("bas_beyin.csv")
bas_beyin.head()


# In[27]:


plt.figure(figsize=(6, 6))
heatmap = sns.heatmap(bas_beyin.corr(), vmin=-1, vmax=1, annot = True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# In[28]:


X = bas_beyin['Bas_cevresi(cm^3)'].values  # Bağımsız Değişken
y= bas_beyin['Beyin_agirligi(gr)'].values  # Bağımlı Değişken
print(X.shape)
print(type(X))


# In[29]:


uzunluk = len(X)
X = X.reshape((uzunluk,1))
print(X.shape)
print(type(X))


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[31]:


from sklearn.linear_model import LinearRegression
modelRegresyon = LinearRegression()
modelRegresyon.fit(X_train, y_train)


# In[32]:


y_pred = modelRegresyon.predict(X_test)


# In[33]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, modelRegresyon.predict(X_train), color = 'blue')
plt.title('Başın Çevre Uzunluğu ve Beyin Ağırlığı (Eğitim Veri Seti)')
plt.xlabel('Başın Çevre Uzunluğu (cm^3)')
plt.ylabel('Beyin Ağırlığı(gram)')
plt.show()


# In[34]:


plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_test, modelRegresyon.predict(X_test), color = 'blue')
plt.title('Başın Çevre Uzunluğu ve Beyin Ağırlığı (Test Veri Seti)')
plt.xlabel('Başın Çevre Uzunluğu (cm^3)')
plt.ylabel('Beyin Ağırlığı(gram)')
plt.show()


# Regresyon denklemi katsayıları ve Regresyon denklemi

# In[35]:


print('Eğim(Q1):', modelRegresyon.coef_)
print('Kesen(Q0):', modelRegresyon.intercept_)
print("y=%0.2f"%modelRegresyon.coef_+"x+%0.2f"%modelRegresyon.intercept_)


# In[36]:


from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score

print("R-Kare: ", r2_score(y_test, y_pred))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print("MedAE: ", median_absolute_error(y_test, y_pred))
print("EVS: ", explained_variance_score(y_test, y_pred))

