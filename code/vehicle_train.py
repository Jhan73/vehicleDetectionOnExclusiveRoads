import object_detection as objDetec

data_yaml = '../configs/config_vehicles_data.yaml'
img_size = (1280, 720 )
model_save_path = './models/plate/runs'

objDetec.train(data_yaml, img_size, epochs=50, save_dir=model_save_path)