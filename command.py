import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Скрипт для запуска обучения или вывода')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Парсер для обучения
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('model_path', help='Путь к модели')
    train_parser.add_argument('data_path', help='Путь к данным')

    # Парсер для вывода
    infer_parser = subparsers.add_parser('infer')
    infer_parser.add_argument('model_path', help='Путь к модели')
    infer_parser.add_argument('input_data', help='Вводные данные')

    # Получаем аргументы
    args = parser.parse_args()

    # Выполняем соответствующую команду
    if args.command == 'train':
        import train
        train.train(args.model_path, args.data_path)
    elif args.command == 'infer':
        import infer
        infer.infer(args.model_path, args.input_data)

if __name__ == "__main__":
    main()
