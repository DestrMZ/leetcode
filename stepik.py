# s = '1:men 2:kind 90:number 0:sun 34:book 56:mountain 87:wood 54:car 3:island 88:power 7:box 17:star 101:ice'
# s = [_.split(':') for _ in s.split()]
# result = dict(s)

# # for i in range(len(s)):
# #     n = 
# result = {int(key): value for i in s.split() for key, value in [i.split(":")]}


# # result = {key[i] for key in range(len(s))}
# print(result)




# student_ids = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010', 'S011', 'S012', 'S013'] 
# student_names = ['Camila Rodriguez', 'Juan Cruz', 'Dan Richards', 'Sam Boyle', 'Batista Cesare', 'Francesco Totti', 'Khalid Hussain', 'Ethan Hawke', 'David Bowman', 'James Milner', 'Michael Owen', 'Gary Oldman', 'Tom Hardy'] 
# student_grades = [86, 98, 89, 92, 45, 67, 89, 90, 100, 98, 10, 96, 93]

# result = [{key: {name: grade}} for key, name, grade in zip(student_ids, student_names, student_grades)]

# print(result)



# my_dict = {'C1': [10, 20, 30, 7, 6, 23, 90], 
#            'C2': [20, 30, 40, 1, 2, 3, 90, 12], 
#            'C3': [12, 34, 20, 21], 
#            'C4': [22, 54, 209, 21, 7], 
#            'C5': [2, 4, 29, 21, 19], 
#            'C6': [4, 6, 7, 10, 55], 
#            'C7': [4, 8, 12, 23, 42], 
#            'C8': [3, 14, 15, 26, 48], 
#            'C9': [2, 7, 18, 28, 18, 28]}


# for i in my_dict.keys():
#     my_dict[i] = [num for num in my_dict[i] if num <= 20]




# emails = {'nosu.edu': ['timyr', 'joseph', 'svetlana.gaeva', 'larisa.mamuk'], 
#           'gmail.com': ['ruslan.chaika', 'rustam.mini', 'stepik-best'], 
#           'msu.edu': ['apple.fruit', 'beegeek', 'beegeek.school'], 
#           'yandex.ru': ['surface', 'google'],
#           'hse.edu': ['tomas-henders', 'cream.soda', 'zivert'],
#           'mail.ru': ['angel.down', 'joanne', 'the.fame.moster']}


# all_emails = []

# for domain, users in emails.items():
#     for user in users:
#         all_emails.append(f'{user}@{domain}')

# all_emails = sorted(all_emails)
# for i in all_emails:
#     print(i)





# transcription_dict = {'G': 'C', 'C': 'G', 'T': 'A', 'A': 'U'}


# result = ''.join(transcription_dict[i] for i in input())

# # for i in user:
# #     result += transcription_dict[i]

# print(result)



# put your python code here

# text = input()

# # Инициализация словаря для подсчета вхождений
# word_counts = {}

# # Разбиваем текст на слова, используя split и удаляем лишние пробелы
# words = text.split()

# # Инициализация списка для хранения порядка вхождений слов
# order_of_appearance = []

# # Подсчет вхождений слов
# for word in words:
#     # Увеличиваем счетчик вхождений слова
#     word_counts[word] = word_counts.get(word, 0) + 1
#     # Добавляем текущее значение счетчика в список вхождений
#     order_of_appearance.append(str(word_counts[word]))

# # Вывод результата: номера вхождения слов на одной строке через пробел
# print(' '.join(order_of_appearance))



# n = int(input())
# # Словарь для хранения прав доступа по каждому файлу
# access_rights = {}

# # Чтение данных о файлах и их правах доступа
# for _ in range(n):
#     entry = input().split()
#     # Имя файла как ключ, множество с правами доступа как значение
#     access_rights[entry[0]] = set(entry[1:])

# # Чтение количества запросов
# m = int(input())

# # Словарь соответствия операций и прав доступа
# operation_to_right = {
#     'execute': 'X',
#     'read': 'R',
#     'write': 'W'
# }

# # Обработка запросов
# for _ in range(m):
#     operation, file_name = input().split()
    
#     # Перевод запроса в требуемое право доступа
#     required_right = operation_to_right.get(operation)
    
#     # Проверка допустимости операции и вывод результата
#     if required_right in access_rights.get(file_name, []):
#         print("OK")
#     else:
#         print("Access denied")




# def merge(dict_list):
#     # Результирующий словарь для объединенных значений
#     result = {}
#     # Проход по всем словарям в списке
#     for dictionary in dict_list:
#         # Проход по всем парам ключ-значение в текущем словаре
#         for key, value in dictionary.items():
#             # Если ключ уже присутствует в результирующем словаре, добавляем значение в множество
#             if key in result:
#                 result[key].add(value)
#             else:
#                 # Иначе создаем новую пару ключ-множество с текущим значением
#                 result[key] = {value}
#     return result





# def build_query_string(params):
#     # Сортировка параметров по ключу и формирование строки запроса
#     return '&'.join(f'{key}={value}' for key, value in sorted(params.items()))




# # put your python code here

# letter_scores = {
#     'A': 1, 'E': 1, 'I': 1, 'L': 1, 'N': 1, 'O': 1, 'R': 1, 'S': 1, 'T': 1, 'U': 1,
#     'D': 2, 'G': 2,
#     'B': 3, 'C': 3, 'M': 3, 'P': 3,
#     'F': 4, 'H': 4, 'V': 4, 'W': 4, 'Y': 4,
#     'K': 5,
#     'J': 8, 'X': 8,
#     'Q': 10, 'Z': 10
# }

# # Ввод слова
# word = input()

# # Подсчет стоимости слова
# total_score = sum(letter_scores[letter] for letter in word)

# # Вывод результата
# print(total_score)





