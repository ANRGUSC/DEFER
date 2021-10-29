import tensorflow as tf

def get_previous(model, name):
    inbound = model.get_layer(name).inbound_nodes[0].inbound_layers
    if type(inbound) != list:
        inbound = [inbound]
    return [layer.name for layer in inbound]

def traverse(model, name, start, part_name, inpt):
    # On subsequent recursive steps, the new input layer will be defined, 
    # so that name needs to be checked in base case
    if (name == start) or (name == part_name):
        return inpt

    output = []
    for n in get_previous(model, name):
        output.append(traverse(model, n, start, part_name, inpt))
    
    # If the DAG node only has 1 previous connection
    if len(output) == 1:
        output = output[0]
    
    layer = model.get_layer(name)
    to_next = layer(output)
    return to_next

def construct_model(model, start, end, part_name="part_begin"):
    inpt = tf.keras.Input(tensor=model.get_layer(start).output, name=part_name)
    output = traverse(model, end, start, part_name, inpt)
    part = tf.keras.Model(inputs=model.get_layer(start).output, outputs=output)
    return part