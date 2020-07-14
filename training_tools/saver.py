import torch


def find_save_dir(args):

    if len(args.misc_args.saved_model_name.strip()) > 0:
        splitted = args.misc_args.saved_model_name.split('/')
        if len(splitted) != 3:
            raise ValueError(f'Invalid saved model specification '
                             f'{args.misc_args.saved_model_name}.\nExpected'
                             '[date]/[exp_name]/checkpoint_iter')
        else:
            saved_date = splitted[0]
            save_exp_name = splitted[1]
            saved_iter = splitted[2]

            actual_save_path = (args.misc_args.heavy_save_dir / '..' / '..'
                                / saved_date
                                / save_exp_name
                                / 'checkpoints'
                                / f'{saved_iter}.pt')

            return actual_save_path

    return None


def save_progress(save_path, iterations, model, optimizer):

    save_path = save_path / f'{iterations}.pt'

    state_dict = {}
    state_dict['model_state_dict'] = model.state_dict()
    state_dict['optimizer_info'] = optimizer.generate_state_dict()

    torch.save(state_dict, save_path.open('wb'))


def load_progress(save_path, model, optimizer, args):
    if not save_path.exists():
        raise ValueError(f'Path {save_path} does not exist.')
    state_dict = torch.load(save_path.open('rb'))

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_info'], args)
