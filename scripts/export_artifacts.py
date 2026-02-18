from ia.exporter import export_db_and_weights

if __name__ == '__main__':
    out = export_db_and_weights()
    print(f'Export creado en: {out}')
