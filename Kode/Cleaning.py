# Initial cleaning the comment data

# Load the Pandas libraries with alias 'pd'
import pandas as pd
from os import listdir
import re
import html.parser

# Get names in dir:
lFiles = listdir(r'Data\Archive')
lFiles.pop()
df = pd.DataFrame()

lFiles = lFiles[40:]

print(str(lFiles[0]) + str(lFiles[-1]))
for f in lFiles:
    dfTemp = pd.read_csv('Data/Archive/' + f)
    df = df.append(dfTemp)

    #dfTemp = pd.read_csv('Data/Archive/' + str(f))
    #df = df.append(f)

df.to_csv('Data/' + '17_11to19_09_raw.csv')

# Read data from file 'filename.csv'
df = pd.read_csv(r'Data\09to14_raw.csv', index_col=0)

# Importing Bot user names
bots = pd.read_csv(r'Data\Bots.csv', index_col=0, sep=';')

# Removing bots from the data
df = df[~df.author.isin(bots.bot_names)]

# Removing any NA's
df.dropna()

# Cleaning the text data, fuld af pis i bunden der prøver hvert enkelt før de røg sammen, slet hvis du ikke er intra
keeplist = "?.!,'_-"

Adj_comment = pd.DataFrame([re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)'
                                   r'[\S]*\s?|(/u/|u/)\S+|(/r/|r/)\S+|[\x00-\x1f\x7f-\xff]|[0-9]+|(&g|&l)\S+'
                                   r'|[^\s\w'+keeplist+']', "", elem) for elem in df['body']], columns=['body'])


df['body'] = Adj_comment['body']


# Hver for sig
'''
text = "Hey, www.reddit.com og http//www.reddit.com og https//www.reddit.com og 134http//www.reddit.com "
text2 = "/u/mitnavn er, u/hallo virker det?,"
text3 = "12/r/12 /r/buttcoin r/suckaniggadickorsummin"
text4 = "I luv my &lt;3 iphone &amp; &gt; &gt &lt; &lt"
text5 = "\b \1 \f"

text = df.iloc[13]

text_ult = "Hey, www.reddit.com og http//www.reddit.com og https//www.reddit.com og 134http//www.reddit.com /u/mitnavn er, u/hallo virker det?,12/r/12 /r/buttcoin r/suckaniggadickorsummin I luv my &lt;3 iphone &amp; &gt; &gt &lt; &lt \b \1 \f"

#result = re.sub(r"http\S+", "", text2) # Starter med http.
# Filter out domains, Also filters out any text not attached to this not separated by a space
re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?','',text)

# Filter out usernames
re.sub(r"(/u/|u/)\S+", "", text2)

# Filter out subreddit
re.sub(r'(/r/|r/)\S+', "", text3)

# Edit HTML, Redigerer html i stedet for at slette det som den gamle gjorde
html.unescape(text4)

# Filter out odd character combinations
# R metoden fjerner egentlig bare &gt; som er html for >, så ikke så umiddelbart brugbar?, jeg fjerner hvertfald både
# > og <, så vi kommer af med: <3
re.sub(r'(&g|&l)\S+', "", text4)


# Filter out control characters, lidt i tvivl om præcis hvad den gør, men noget med ting markeret med \
re.sub(r'[\x00-\x1f\x7f-\xff]', '', text5)

# Filter out numbers
re.sub(r'[0-9]+', '', text3)
'''
