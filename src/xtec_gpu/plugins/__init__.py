from . import cluster_data


def plugin_menu():
    menu = "XTEC-GPU"
    actions = [(("Cluster Data", cluster_data.show_dialog))]

    return menu, actions
