import pandas as pd
from Models.mcdm import Cv

data = pd.DataFrame([[1.0, 0.9, 1.0],
                     [1.0, 0.9, 0.8],
                     [5.1, 0.9, 1.0]], columns=list('ABC'))
model = Cv()
model.fit(data)

print(model.weight)
print(model.info)
print(model.test())

