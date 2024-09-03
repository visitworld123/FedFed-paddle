
import paddle



def create_optimizer(args, model=None, params=None, **kwargs):
    if "role" in kwargs:
        role = kwargs["role"]
    else:
        role = args.role

    # params has higher priority than model
    if params is not None:
        params_to_optimizer = params
    else:
        if model is not None:
            params_to_optimizer = model.parameters()
        else:
            raise NotImplementedError
        pass

    if (role == 'server') and (args.algorithm in [
        'FedAvg']):
        if args.server_optimizer == "sgd":
            # optimizer = paddle.optimizer.SGD(params_to_optimizer,
            #     learning_rate=args.lr, weight_decay=args.wd, momentum=args.momentum, use_nesterov=args.nesterov)
            optimizer = paddle.optimizer.SGD(filter(lambda p: p.requires_grad, params_to_optimizer),
                learning_rate=args.lr, weight_decay=args.wd, momentum=args.momentum, use_nesterov=args.nesterov)
        elif args.server_optimizer == "adam":
            optimizer = paddle.optimizer.Adam(filter(lambda p: p.requires_grad, params_to_optimizer),
                learning_rate=args.lr, weight_decay=args.wd, amsgrad=True)
        elif args.server_optimizer == "no":
            optimizer = paddle.optimizer.Momentum(parameters=filter(lambda p: not p.stop_gradient, params_to_optimizer),
                learning_rate=args.lr, weight_decay=args.wd, momentum=args.momentum, use_nesterov=args.nesterov)
        else:
            raise NotImplementedError
    else:
        if args.client_optimizer == "sgd":
            optimizer = paddle.optimizer.Momentum(params_to_optimizer,
                learning_learning_raterate=args.lr, weight_decay=args.wd, momentum=args.momentum, use_nesterov=args.nesterov)
        elif args.client_optimizer == "adam":
            raise NotImplementedError
        elif args.client_optimizer == "no":
            optimizer = paddle.optimizer.Momentum(parameters=params_to_optimizer,
                learning_rate=args.lr, weight_decay=args.wd, momentum=args.momentum, use_nesterov=args.nesterov)
        else:
            raise NotImplementedError

    return optimizer







