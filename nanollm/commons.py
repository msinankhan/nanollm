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
        home_dir=os.path.expanduser("/disk2")
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
        if os.path.exists(lock_path):
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
