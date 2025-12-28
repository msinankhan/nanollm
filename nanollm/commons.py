import os
import torch
import torch.distributed as dist
import re
import logging
import urllib.request
from filelock import FileLock

class ColoredFormatter(logging.Formatter):
    "Adds colors to the log messages"
    COLORS={
        'INFO':  '\033[32m',
        'ERROR':  '\033[31m',
        'DEBUG':  '\033[36m',
        'WARNIGN': '\033[33m',
        'CRITICAL': '\033[35m',
    }

    RESET= '\033[0m'
    BOLD=  '\033[1m'
    def format(self,record):
        levelname=record.levelname

        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{self.BOLD}{levelname}{self.RESET}"

        message=super().format(record)

        if levelname=='INFO':
            message=re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}',message)
            message=re.sub(r'(Shard \d+)',rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.REST}',message)
        return message
    
def setup_default_logging():
    handler=logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig (
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger=logging.getLogger(__name__)


def get_base_dir():
    "Get nanollm's base directory"

    if os.environ.get("NANOLLM_BASE_DIR"):
        nanollm_dir=os.environ.get('NANOLLM_BASE_DIR')
    else:
        home_dir=os.path.expanduser("~")
        cache_dir=os.path.join(home_dir, ".cache")
        nanollm_dir=os.path.join(cache_dir,"nanollm")

    os.makedirs(nanollm_dir,exist_ok=True)
    return nanollm_dir


def download_data_with_lock(url, filename, postprocess_fn=None):
    "Download the data from url and optional postprocess."


    base_dir=get_base_dir()
    file_path=os.path.join(base_dir,filename)
    lock_path=file_path + ".lock"


    if os.path.exists(file_path):
        return file_path
    
    with FileLock(lock_path):
        if os.path.exists(file_path):
            return file_path
        
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content=response.read()

        with open(file_path,'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")
    
        if postprocess_fn is not None:
            postprocess_fn(file_path)
    
    return file_path


def print0(s="",**kwargs):
    """
    The RANK environment variable is set by torchrun when launching distributed training with multiple GPUs

    This pattern ensures clean, non-duplicated logging output in distributed training scenarios
    """
    ddp_rank=int(os.environ.get('RANK',0))

    if ddp_rank==0:
        print(s,**kwargs)

def print_banner():
    banner= """

        ****     **                              **  **            
        /**/**   /**                             /** /**            
        /**//**  /**  ******   *******   ******  /** /** ********** 
        /** //** /** //////** //**///** **////** /** /**//**//**//**
        /**  //**/**  *******  /**  /**/**   /** /** /** /** /** /**
        /**   //**** **////**  /**  /**/**   /** /** /** /** /** /**
        /**    //***//******** ***  /**//******  *** *** *** /** /**
        //      ///  //////// ///   //  //////  /// /// ///  //  // 

        """
    print0(banner) 

def is_ddp_requested() -> bool:
    return all(k in os.environ for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))

def is_ddp_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_requested():
        assert all (var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])

        ddp_rank=int(os.environ.get('RANK'))
        ddp_local_rank=int(os.environ['LOCAL_RANK'])
        ddp_world_size=int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False,0,0,1
    

def detect_device_type():
    if torch.cuda.is_available():
        device_type='cuda'
    else:
        device_type='cpu'
    print0(f"Auto Detected device_type {device_type}")

def compute_init(device_type='cuda'):

    assert device_type in ["cuda", "cpu"], "Unsuported Device Type"

    if device_type=='cuda':
        assert torch.cuda.is_available() , "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"


    # PyTorch has two RNGs:
    # CPU RNG
    # CUDA RNG
    torch.manual_seed(42)
    if device_type=='cuda':
        torch.cuda.manual_seed(42) #CUDA has its own RNG, separate from the CPU RNG. Without this:
                                   # CPU and GPU randomness diverge
                                   # Different ranks may initialize differently

    

    if device_type=='cuda':
        torch.backends.cuda.matmul.fp32_precision='tf32'


    is_ddp_requested, ddp_rank,ddp_local_rank,ddp_world_size= get_dist_info()

    if is_ddp_initialized and device_type=='cuda':
        device=torch.device('cuda', ddp_rank) #Initializes as cuda:0, cuda:1, cuda:2 etc
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl',device_id=device) #Creates communication channels between processes
        dist.barrier() #This forces all processes to wait until everyone has finished initialization

    else:
        torch.device(device_type)

    if ddp_rank==0:
        logger.info(f"Distribute world size: {ddp_world_size}")


    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    if is_ddp_initialized():
        dist.destroy_process_group()

