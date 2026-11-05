import pandas as pd

def convert_to_sentences(csv_file):
    df = pd.read_csv(csv_file)

    sentences = []

    # grouping tokens belonging to the same sentence
    for sentence_id, group in df.groupby('sentence_id'):
        # placing tokens in the correct order
        group = group.sort_values('word_index')

        words = group['word'].tolist()
        tags = group['tag'].tolist()

        sentence_text = ' '.join(words)

        entities = []

        current_entity = []
        current_type = None

        for word, tag in zip(words, tags):
            if tag == 'O':
                # completing any existing entity 
                if current_entity:
                    entities.append({
                        'text' : ' '.join(current_entity),
                        'label' : current_type
                    })

                    current_entity = []
                    current_type = None

            elif tag.startswith('B-'):
                # finishing previous entity 
                if current_entity:
                    entities.append({
                        'text' : ' '.join(current_entity),
                        'label' : current_entity
                    })

                # starting new entity 
                current_entity = [word]
                current_type = tag[2:]

            elif tag.startswith('I-'):
                # continue the current entity 
                current_entity.append(word)

        # finishing entity at the end of the sentence 
        if current_entity:
            entities.append({
                'text' : ' '.join(current_entity),
                'label' : current_type
            })

        sentences.append({
            'sentence_id' : sentence_id,
            'text' : sentence_text,
            'entities' : entities
        })
    return sentences

if __name__ == '__main__':
    file = "anonymous_telner/dataset/dataset-generic.csv"

    sentences = convert_to_sentences(file)

    print('number of sentences :', len(sentences))

    print('\nfirst 5 sentences :\n')

    for item in sentences[:5]:
        print('='*60)
        print('sentence id :', item['sentence_id'])
        print('text :', item['text'])
        print('enitites :')

        for entity in item['entities']:
            print('\t', entity)