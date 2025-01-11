#!/usr/bin/env python
# coding: utf-8

# In[1]:


# For beta versions: `pip install --pre -U "weaviate-client==4.*"`
#get_ipython().system('pip install -U weaviate-client')


# In[2]:


import pandas as pd
import sentence_transformers
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from weaviate.collections.classes.filters import Filter
from tqdm import tqdm
from typing_extensions import Annotated, deprecated


# In[3]:




# In[4]:


file_path = '/data/terzine.csv'
df = pd.read_csv(file_path, delimiter=';')
df


# # Nuova sezione

# In[5]:


#model = SentenceTransformer('sentence-transformers/LaBSE')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(df.terzina.tolist(), show_progress_bar=True)


# In[6]:


df["embedding"] = list(embeddings)


# In[7]:


df


# In[30]:


df.to_parquet('/data/terzine_vectors.parquet')


# In[9]:


df


# In[10]:


WEAVIATE_URL = "https://rabbfo5drdqk3qa0lpjpvq.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY  = "KIqaLabClZWRjtyT9G10ki3FhzqETU6x81yv"
collection_name = "Voci_dall_Inferno"

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
)

if client.collections.exists(collection_name):  # In case we've created this collection before
    client.collections.delete(collection_name)  # THIS WILL DELETE ALL DATA IN THE COLLECTION

voci_dall_inferno = client.collections.create(
    name=collection_name,
    properties=[
        wvc.config.Property(
            name="cantica",
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="canto",
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="range_versi",
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="terzina",
            data_type=wvc.config.DataType.TEXT
        ),
    ]
)

print(client.is_ready())


# In[11]:


# Prepare all the data rows first
data_rows = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing data"):
    data_rows.append({
        "properties": {
            "cantica": row['cantica'],
            "canto": row['canto'],
            "range_versi": row['range_versi'],
            "terzina": row['terzina']
            #"chapter": int(row['chapter']),
            #"verse": int(row['verse'])
        },
        "vector": row['embedding']
    })


# In[12]:


# Now perform the batch insertion
with voci_dall_inferno.batch.dynamic() as batch:
    for data_row in tqdm(data_rows, desc="Inserting data"):
        batch.add_object(
            properties=data_row['properties'],
            vector=data_row['vector']
        )


# In[13]:


from weaviate.classes.query import MetadataQuery


# In[14]:


#query = "Si quis uult post me uenire, abneget semetipsum et tollat crucem suam et sequatur me"
query = "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura che la diritta via era smarrita"
#def find_similar(query, threshold):
def find_similar(query):
    query_vector = model.encode([query])[0]
    response = voci_dall_inferno.query.near_vector(
        near_vector=query_vector,
        limit=10,
        return_metadata=MetadataQuery(distance=True)
    )

    for o in response.objects:
        #if o.metadata.distance < threshold:
            print(o.properties["canto"], o.properties["range_versi"], ": ", o.properties["terzina"])
            print(o.metadata.distance)


# In[ ]:





# In[15]:


query = "vuolsi così colà dove si puote ciò che si vuole"
#find_similar(query, threshold=.4)
find_similar(query)


# In[ ]:









# In[16]:


text = """
Così discesi del cerchio primaio
giù nel secondo, che men loco cinghia,
e tanto più dolor, che punge a guaio.                             3

Stavvi Minòs orribilmente, e ringhia:
essamina le colpe ne l’intrata;
giudica e manda secondo ch’avvinghia.                        6

Dico che quando l’anima mal nata
li vien dinanzi, tutta si confessa;
e quel conoscitor de le peccata                                       9

vede qual loco d’inferno è da essa;
cignesi con la coda tante volte
quantunque gradi vuol che giù sia messa.                  12

Sempre dinanzi a lui ne stanno molte;
vanno a vicenda ciascuna al giudizio;
dicono e odono, e poi son giù volte.                              15

«O tu che vieni al doloroso ospizio»,
disse Minòs a me quando mi vide,
lasciando l’atto di cotanto offizio,                                    18

«guarda com’entri e di cui tu ti fide;
non t’inganni l’ampiezza de l’intrare!».
E ’l duca mio a lui: «Perché pur gride?                         21

Non impedir lo suo fatale andare:
vuolsi così colà dove si puote
ciò che si vuole, e più non dimandare».                       24

Or incomincian le dolenti note
a farmisi sentire; or son venuto
là dove molto pianto mi percuote.                                  27

Io venni in loco d’ogne luce muto,
che mugghia come fa mar per tempesta,
se da contrari venti è combattuto.                                  30

La bufera infernal, che mai non resta,
mena li spirti con la sua rapina;
voltando e percotendo li molesta.                                  33

Quando giungon davanti a la ruina,
quivi le strida, il compianto, il lamento;
bestemmian quivi la virtù divina.                                    36

Intesi ch’a così fatto tormento
enno dannati i peccator carnali,
che la ragion sommettono al talento.                            39

E come li stornei ne portan l’ali
nel freddo tempo, a schiera larga e piena,
così quel fiato li spiriti mali;                                             42

di qua, di là, di giù, di sù li mena;
nulla speranza li conforta mai,
non che di posa, ma di minor pena.                              45

E come i gru van cantando lor lai,
faccendo in aere di sé lunga riga,
così vid’io venir, traendo guai,                                         48

ombre portate da la detta briga;
per ch’i’ dissi: «Maestro, chi son quelle
genti che l’aura nera sì gastiga?».                                 51

«La prima di color di cui novelle
tu vuo’ saper», mi disse quelli allotta,
«fu imperadrice di molte favelle.                                     54

A vizio di lussuria fu sì rotta,
che libito fé licito in sua legge,
per tòrre il biasmo in che era condotta.                         57

Ell’è Semiramìs, di cui si legge
che succedette a Nino e fu sua sposa:
tenne la terra che ’l Soldan corregge.                            60

L’altra è colei che s’ancise amorosa,
e ruppe fede al cener di Sicheo;
poi è Cleopatràs lussuriosa.                                           63

Elena vedi, per cui tanto reo
tempo si volse, e vedi ’l grande Achille,
che con amore al fine combatteo.                                  66

Vedi Parìs, Tristano»; e più di mille
ombre mostrommi e nominommi a dito,
ch’amor di nostra vita dipartille.                                      69

Poscia ch’io ebbi il mio dottore udito
nomar le donne antiche e ’ cavalieri,
pietà mi giunse, e fui quasi smarrito.                            72

I’ cominciai: «Poeta, volontieri
parlerei a quei due che ’nsieme vanno,
e paion sì al vento esser leggeri».                                 75

Ed elli a me: «Vedrai quando saranno
più presso a noi; e tu allor li priega
per quello amor che i mena, ed ei verranno».             78

Sì tosto come il vento a noi li piega,
mossi la voce: «O anime affannate,
venite a noi parlar, s’altri nol niega!».                            81

Quali colombe dal disio chiamate
con l’ali alzate e ferme al dolce nido
vegnon per l’aere dal voler portate;                                84

cotali uscir de la schiera ov’è Dido,
a noi venendo per l’aere maligno,
sì forte fu l’affettuoso grido.                                              87

«O animal grazioso e benigno
che visitando vai per l’aere perso
noi che tignemmo il mondo di sanguigno,                  90

se fosse amico il re de l’universo,
noi pregheremmo lui de la tua pace,
poi c’hai pietà del nostro mal perverso.                        93

Di quel che udire e che parlar vi piace,
noi udiremo e parleremo a voi,
mentre che ’l vento, come fa, ci tace.                             96

Siede la terra dove nata fui
su la marina dove ’l Po discende
per aver pace co’ seguaci sui.                                        99

Amor, ch’al cor gentil ratto s’apprende
prese costui de la bella persona
che mi fu tolta; e ’l modo ancor m’offende.                 102

Amor, ch’a nullo amato amar perdona,
mi prese del costui piacer sì forte,
che, come vedi, ancor non m’abbandona.                  105

Amor condusse noi ad una morte:
Caina attende chi a vita ci spense».
Queste parole da lor ci fuor porte.                                108

Quand’io intesi quell’anime offense,
china’ il viso e tanto il tenni basso,
fin che ’l poeta mi disse: «Che pense?».                    111

Quando rispuosi, cominciai: «Oh lasso,
quanti dolci pensier, quanto disio
menò costoro al doloroso passo!».                              114

Poi mi rivolsi a loro e parla’ io,
e cominciai: «Francesca, i tuoi martìri
a lagrimar mi fanno tristo e pio.                                     117

Ma dimmi: al tempo d’i dolci sospiri,
a che e come concedette Amore
che conosceste i dubbiosi disiri?».                              120

E quella a me: «Nessun maggior dolore
che ricordarsi del tempo felice
ne la miseria; e ciò sa ’l tuo dottore.                            123

Ma s’a conoscer la prima radice
del nostro amor tu hai cotanto affetto,
dirò come colui che piange e dice.                               126

Noi leggiavamo un giorno per diletto
di Lancialotto come amor lo strinse;
soli eravamo e sanza alcun sospetto.                         129

Per più fiate li occhi ci sospinse
quella lettura, e scolorocci il viso;
ma solo un punto fu quel che ci vinse.                         132

Quando leggemmo il disiato riso
esser basciato da cotanto amante,
questi, che mai da me non fia diviso,                           135

la bocca mi basciò tutto tremante.
Galeotto fu ’l libro e chi lo scrisse:
quel giorno più non vi leggemmo avante».                 138

Mentre che l’uno spirto questo disse,
l’altro piangea; sì che di pietade
io venni men così com’io morisse.

E caddi come corpo morto cade.
"""


# In[17]:


import spacy


# In[18]:


nlp = spacy.blank("it")
nlp.add_pipe("sentencizer")


# In[19]:


sentences = [sent.text for sent in nlp(text).sents]


# In[20]:


len(sentences)


# In[21]:


for sent in sentences:
    query_vector = model.encode([sent])[0]
    response = voci_dall_inferno.query.near_vector(
        near_vector=query_vector,
        limit=10,
        return_metadata=MetadataQuery(distance=True)
    )

    for o in response.objects:
        if o.metadata.distance < .4:
            print(F"MATCH: {sent}")
            print(o.metadata.distance)
            print(o.properties)


# In[22]:


#get_ipython().system('pip install streamlit')


# In[23]:


#get_ipython().system("python '/data/streamlit_app_terzine_3.py'")
'/data/streamlit_app_terzine_3.py'
