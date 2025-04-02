from itertools import product

class QueryExpander:

    def __init__(self, synonym_list):
        self.synonym_list = synonym_list
        self.inverse = {}

        self.__invert_synonyms_dict__()

    def __invert_synonyms_dict__(self):
        for key, values in self.synonym_list.items():
            for value in values:
                aux = values.copy()
                aux.remove(value)
                self.inverse[value] = [key] + aux

    def __find_words__(self, query: str):
      """
      encuentra todas las palabras simples y compuestas en la consulta que están
      en el diccionario de sinónimos
      """
      preprocessed_query = (query.replace(',', ' , ').replace('¿', '¿ ')
                            .replace('?', ' ?').replace('.', ' . ').replace('(', ' ( ')
                            .replace(')', ' ) '))
      words = preprocessed_query.lower().split()
      words_found = []
      found = False
      i = 0

      while i < len(words):
        """
        intentar encontrar la palabra compuesta más larga posible
        asumimos que como máximo está formada por 5 palabras
        """
        for j in range(min(5, len(words) - i), 0, -1):
          if not found:
              compound_word = ' '.join(words[i:i + j])
              if compound_word in self.synonym_list or compound_word in self.inverse:
                words_found.append(compound_word)
                #saltamos las palabras ya procesadas
                i += j
                found = True
        if not found:
          words_found.append(words[i])
          i += 1
        found = False

      return words_found


    def query_expansion(self, query: str):
        """
        genera todas las combinaciones posibles de la consulta sustituyendo las palabras encontradas
        por sus sinónimos
        """
        words_found = self.__find_words__(query)
        aux_list = []
        for word in words_found:
            if word in self.synonym_list.keys():
                elem = [word] + self.synonym_list[word]
            elif word in self.inverse.keys():
                elem = [word] + self.inverse[word]
            else:
                elem = [word]
            aux_list.append(elem)
        #genera todas las posibles combinaciones calculando el producto cartesiano
        return list(product(*aux_list))