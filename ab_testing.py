import warnings
warnings.filterwarnings('ignore')
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)




############################
# Görev 1: Veriyi Hazırlama ve Analiz Etme
############################


df_c = pd.read_excel("/Users/rmarabaci/PycharmProjects/pythonProject1/data_science_bootcamp/Hafta_5_Measurement_Problems/odev_2_AB_testing/ab_testing.xlsx" , sheet_name= 'Control Group')
df_c.describe().T

df_t = pd.read_excel('/Users/rmarabaci/PycharmProjects/pythonProject1/data_science_bootcamp/Hafta_5_Measurement_Problems/odev_2_AB_testing/ab_testing.xlsx', sheet_name= 'Test Group')
df_t.describe().T

check_df(df_t)


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df_c)
check_df(df_t)

df_c.columns = [col + '_control' for col in df_c ]
df_t.columns = [col + '_test' for col in df_t ]

df = pd.concat([df_c, df_t] , axis=1)

df.head()


############################
# Görev 2: A/B Testinin Hipotezinin Tanımlanması
############################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman(false?) girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


# H0 : M1 = M2 (Iki grup arasinda tıklanan reklamlar sonrası kazanclar baglaminda istatistiksel olarak anlamli bir fark yoktur.)
# H1 : M1!= M2 (... vardir)


df['Purchase_control'].mean()
df['Purchase_test'].mean()

############################
# Görev 3: Hipotez Testinin Gerçekleştirilmesi
############################

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# Bunlar Normallik Varsayımı ve Varyans Homojenliğidir. Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni
# üzerinden ayrı ayrı test ediniz.
#
# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ? Elde edilen p-value değerlerini yorumlayınız.


#shapiro testi, bir degiskenin dagiliminin normal olup olmadigini test eder.

test_stat, pvalue = shapiro(df['Purchase_control'])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9773, p-value = 0.5891

test_stat, pvalue = shapiro(df['Purchase_test'])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9589, p-value = 0.1541

#Iki grup icin de p > 0.05 oldugu icin H0 REDDEDİLEMEZ. Iki deger de normal dagilmaktadir.


# Varyans Homojenliği :
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
# Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.

test_stat, pvalue = levene(df['Purchase_control'],
                           df['Purchase_test'])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 2.6393, p-value = 0.1083
# p > 0.05 oldugu icin H0 REDDEDİLEMEZ. Bu sebeple varyanslar homojendir.

#
# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.

#Normallik Varsayımı ve Varyans Homojenliği sağlandigi icin bağımsız iki örneklem t testi (parametrik test) kullanacagiz

test_stat, pvalue = ttest_ind(df['Purchase_control'],
                               df['Purchase_test'],
                                equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = -0.9416, p-value = 0.3493


# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak
# kontrol ve test grubu satın alma ortalamaları arasında istatistiki
# olarak anlamlı bir fark olup olmadığını yorumlayınız.

# Test Stat = -0.9416, p-value = 0.3493

# p-value>0.05 oldugu icin H0 reddedilemez.

# kontrol ve test grubu satın alma ortalamaları arasında istatistiki
# olarak anlamlı bir fark yoktur.


############################
# Görev 4: Sonuçların Analizi
############################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

#shapiro testi, bir degiskenin dagiliminin normal olup olmadigini test eder.
# Bu sebeple normal dagilim kontrolu icin bu testi kullandim.

# varyans hipotezlerinin kontrolu icin levene testi kullanilir.
# Bu sebeple varyans hipotezi kontrolu icin bu testi kullandim.

#Normallik Varsayımı ve Varyans Homojenliği sağlandigi
# icin bağımsız iki örneklem t testi (parametrik test) kullanilir.
# Bu sebeple bu testi kullandim.


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.
# Iki yontem arasinda istatistiki olarak anlamli bir fark bulunamamistir.
# O sebeple teklif turlerini degistirmek icin musterinin ekstra bir masraf yapmasina gerek olmadigi ortaya cikmistir.




