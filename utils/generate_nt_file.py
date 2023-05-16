from pathlib import Path
from utils.util import load_pickle, write_lines_to_file, get_id_map, map_id_to_name_in_query


def modify_name_to_iri(name, prefix=True):
    if '@' in name:
        name = name[1:name.find('@') - 1]
    if '[' in name:
        name = name[1:-1]
    if prefix:
        iri = '<http://rdf.freebase.com/' + name + '>'
    else:
        iri = '<fb:' + name + '>'
    return iri.replace(' ', '_')


def generate_nt_file(dataset_path=Path('./data/FB15k-237-betae')):
    def get_nt_lines(file_path):
        nt_lines = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                id1, id2, id3 = [int(_) for _ in line.split()]
                s = id2name[id1]
                p = id2rel[id2]
                o = id2name[id3]

                s = id2ent[id1] if s == 'null' else s[1:-4]
                p = id2ent[id2] if p == 'null' else p
                o = id2ent[id3] if o == 'null' else o[1:-4]

                # for id, name in zip((id1, id2, id3), (s, p, o)):
                #     # these entities are placed by new entities
                #     # so they don't have name
                #     if name == 'null':
                #         ent_missing_name.append(id2ent[id])

                nt_lines.append(
                    '\t'.join([modify_name_to_iri(s), modify_name_to_iri(p), modify_name_to_iri(o), '.', '\n']))

        return nt_lines

    id2ent_path = dataset_path / 'id2ent.pkl'
    id2ent = load_pickle(id2ent_path)

    id2name_path = dataset_path / 'id2name.pkl'
    id2name = load_pickle(id2name_path)

    id2rel_path = dataset_path / 'id2rel.pkl'
    id2rel = load_pickle(id2rel_path)

    train_path = dataset_path / 'train.txt'
    train_nt_lines = get_nt_lines(train_path)

    valid_path = dataset_path / 'valid.txt'
    valid_nt_lines = get_nt_lines(valid_path)

    test_path = dataset_path / 'test.txt'
    test_nt_lines = get_nt_lines(test_path)

    train_nt_path = Path(train_path).parent / 'G_train.nt'
    write_lines_to_file(train_nt_path, train_nt_lines)

    test_nt_path = Path(train_path).parent / 'G_test.nt'
    write_lines_to_file(test_nt_path, train_nt_lines + valid_nt_lines + test_nt_lines)


generate_nt_file()


def convert_name_to_iri_in_query(query, prefix=True):
    for i, item in enumerate(query):
        # print(item)
        if type(item) is list:
            query[i] = convert_name_to_iri_in_query(item, prefix)
        else:
            query[i] = modify_name_to_iri(item, prefix)
    return query


if __name__ == '__main__':
    # generate_nt_file()
    data_dir = Path('./data/FB15k-237-betae/')

    queries = load_pickle(data_dir / 'test-queries.pkl')
    answers1 = load_pickle(data_dir / 'test-easy-answers.pkl')
    answers2 = load_pickle(data_dir / 'test-hard-answers.pkl')

    id2ent, id2name, id2rel, id2type = get_id_map('./data/FB15k-237-betae/')

    for k in queries:
        query = queries[k].pop()

        print('easy')
        for id in answers1[query]:
            print(id2name[id])
        print('hard')
        for id in answers2[query]:
            print(id2name[id])

        query = map_id_to_name_in_query(query, id2name, id2rel)

        print(convert_name_to_iri_in_query(query))
