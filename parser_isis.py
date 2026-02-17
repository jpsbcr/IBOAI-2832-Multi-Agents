def parse_isis_config(config_text):
    """
    Convierte el output plano de 'show run router isis' en un diccionario jerárquico.
    Mantiene la estructura según la indentación.
    """
    lines = config_text.strip().splitlines()
    root = {}
    stack = [(root, -1)]  # (dict_actual, nivel_indentación)

    for line in lines:
        line = line.rstrip()
        if not line or line.strip() in ("!",):
            continue
        if line.startswith("Sun "):
            continue  # ignorar timestamp

        indent = len(line) - len(line.lstrip())
        command = line.strip()

        # Retroceder en la jerarquía si la indentación es menor
        while stack and indent <= stack[-1][1]:
            stack.pop()

        current_dict = stack[-1][0]

        # Si la siguiente línea es más indentada, creamos un sub-bloque
        # Detectamos esto viendo el próximo elemento en la lista
        if not any(command == k for k in current_dict):
            current_dict[command] = {}
        
        stack.append((current_dict[command], indent))

    return root
