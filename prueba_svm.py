import pandas as pd
import pickle


ruta_csv = 'C:/Users/Usuario/Desktop/COMMENTS_GAL/'

ruta_pkl = 'C:/Users/Usuario/Desktop/COMMENTS_GAL/best_model_SVM.pkl'


data = pd.read_csv(ruta_csv)


with open(ruta_pkl, 'rb') as model_file:
    model = pickle.load(model_file)


predictions = model.predict(data['text'])


# Opcional: agregar las predicciones al DataFrame original
data['predicciones'] = predictions

# Guardar el DataFrame con las predicciones a un nuevo archivo CSV si es necesario
output_file_path = 'ruta_a_tu_archivo_con_predicciones.csv'
data.to_csv(output_file_path, index=False)

print("Predicciones realizadas y guardadas en el archivo:", output_file_path)


