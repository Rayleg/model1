Этот классификатор решает задачу классификации по полу
Инструкция:
columns = trees.choose_targets()
	команда вернет список признаков по умолчанию, по которому пройдет обучение
x, y, names = trees.read_data_set(columns=columns)
	команда вернет значения признаков для обучения, цели, названия признаков
trees.print_set_stats(x, y, names)
	все это можно вывести
cls = trees.build_classifier()
	строится классификатор с параметрами по умолчанию, они выводятся
	на показе работ был такой классификатор: build_classifier(0.001, 5, 100)
cls = cls.fit(x, y)
	обучается классификатор
cls.tree.draw_me(names)
	можно вывести дерево решений в файл по умолчанию
print mx.f1_score(y, cls.predict(x), average='micro')
	можно вывести какую-нибудь оценку из sklearn.metrics
	
Доступны predict, pruning
