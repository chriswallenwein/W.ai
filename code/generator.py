import numpy as np

class Generate():

    @staticmethod
    def random_x(x1_mean, x1_std, x2_mean, x2_std, n):
        random_generator = np.random.default_rng()
        x1 = random_generator.normal(x1_mean,x1_std, size=n)
        x2 = random_generator.normal(x2_mean,x2_std, size=n)
        data = np.array([x1,x2]).T
        return data
    
    @staticmethod
    def random_classes(number_of_classes, samples_per_class):
        random_generator = np.random.default_rng()
        x = []
        
        for _ in range(number_of_classes):
            x1_mean = random_generator.integers(number_of_classes * 10)
            x2_mean = random_generator.integers(number_of_classes * 10)
            
            x1_std = random_generator.integers(1, number_of_classes)
            x2_std = random_generator.integers(1, number_of_classes)
            
            generated_data = Generate.random_x(x1_mean, x1_std, x2_mean, x2_std, samples_per_class)
            x.append(generated_data)
        
        x = np.concatenate(x)
        
        classes = np.arange(number_of_classes)
        y = np.repeat(classes, samples_per_class)
        
        data = (x, y)
        return data