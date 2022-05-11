df['pseudonym'] = df['username'].map(
        lambda x: fakes.user_name())
df['pseudonym'].head()
