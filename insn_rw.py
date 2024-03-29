import os, pickle
from time import time
from typing import List
import idc
import idaapi
import idautils
import ida_pro
import ida_auto
import ida_nalt


from capstone import *
from capstone.x86 import *
md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True


def get_eflag_name(eflag):
    if eflag == X86_EFLAGS_UNDEFINED_OF:
        return "UNDEF_OF"
    elif eflag == X86_EFLAGS_UNDEFINED_SF:
        return "UNDEF_SF"
    elif eflag == X86_EFLAGS_UNDEFINED_ZF:
        return "UNDEF_ZF"
    elif eflag == X86_EFLAGS_MODIFY_AF:
        return "MOD_AF"
    elif eflag == X86_EFLAGS_UNDEFINED_PF:
        return "UNDEF_PF"
    elif eflag == X86_EFLAGS_MODIFY_CF:
        return "MOD_CF"
    elif eflag == X86_EFLAGS_MODIFY_SF:
        return "MOD_SF"
    elif eflag == X86_EFLAGS_MODIFY_ZF:
        return "MOD_ZF"
    elif eflag == X86_EFLAGS_UNDEFINED_AF:
        return "UNDEF_AF"
    elif eflag == X86_EFLAGS_MODIFY_PF:
        return "MOD_PF"
    elif eflag == X86_EFLAGS_UNDEFINED_CF:
        return "UNDEF_CF"
    elif eflag == X86_EFLAGS_MODIFY_OF:
        return "MOD_OF"
    elif eflag == X86_EFLAGS_RESET_OF:
        return "RESET_OF"
    elif eflag == X86_EFLAGS_RESET_CF:
        return "RESET_CF"
    elif eflag == X86_EFLAGS_RESET_DF:
        return "RESET_DF"
    elif eflag == X86_EFLAGS_RESET_IF:
        return "RESET_IF"
    elif eflag == X86_EFLAGS_TEST_OF:
        return "TEST_OF"
    elif eflag == X86_EFLAGS_TEST_SF:
        return "TEST_SF"
    elif eflag == X86_EFLAGS_TEST_ZF:
        return "TEST_ZF"
    elif eflag == X86_EFLAGS_TEST_PF:
        return "TEST_PF"
    elif eflag == X86_EFLAGS_TEST_CF:
        return "TEST_CF"
    elif eflag == X86_EFLAGS_RESET_SF:
        return "RESET_SF"
    elif eflag == X86_EFLAGS_RESET_AF:
        return "RESET_AF"
    elif eflag == X86_EFLAGS_RESET_TF:
        return "RESET_TF"
    elif eflag == X86_EFLAGS_RESET_NT:
        return "RESET_NT"
    elif eflag == X86_EFLAGS_PRIOR_OF:
        return "PRIOR_OF"
    elif eflag == X86_EFLAGS_PRIOR_SF:
        return "PRIOR_SF"
    elif eflag == X86_EFLAGS_PRIOR_ZF:
        return "PRIOR_ZF"
    elif eflag == X86_EFLAGS_PRIOR_AF:
        return "PRIOR_AF"
    elif eflag == X86_EFLAGS_PRIOR_PF:
        return "PRIOR_PF"
    elif eflag == X86_EFLAGS_PRIOR_CF:
        return "PRIOR_CF"
    elif eflag == X86_EFLAGS_PRIOR_TF:
        return "PRIOR_TF"
    elif eflag == X86_EFLAGS_PRIOR_IF:
        return "PRIOR_IF"
    elif eflag == X86_EFLAGS_PRIOR_DF:
        return "PRIOR_DF"
    elif eflag == X86_EFLAGS_TEST_NT:
        return "TEST_NT"
    elif eflag == X86_EFLAGS_TEST_DF:
        return "TEST_DF"
    elif eflag == X86_EFLAGS_RESET_PF:
        return "RESET_PF"
    elif eflag == X86_EFLAGS_PRIOR_NT:
        return "PRIOR_NT"
    elif eflag == X86_EFLAGS_MODIFY_TF:
        return "MOD_TF"
    elif eflag == X86_EFLAGS_MODIFY_IF:
        return "MOD_IF"
    elif eflag == X86_EFLAGS_MODIFY_DF:
        return "MOD_DF"
    elif eflag == X86_EFLAGS_MODIFY_NT:
        return "MOD_NT"
    elif eflag == X86_EFLAGS_MODIFY_RF:
        return "MOD_RF"
    elif eflag == X86_EFLAGS_SET_CF:
        return "SET_CF"
    elif eflag == X86_EFLAGS_SET_DF:
        return "SET_DF"
    elif eflag == X86_EFLAGS_SET_IF:
        return "SET_IF"
    else: 
        return None

def getCapstoneInsFLAG(insn):
    # "data" instruction generated by SKIPDATA option has no detail
    if insn.id == 0:
        return "NOTDEFINED"

    if insn.eflags:
        updated_flags = []
        for i in range(0,46):
            if insn.eflags & (1 << i):
                updated_flags.append(get_eflag_name(1 << i))
        EFLAGS = str(','.join(p for p in updated_flags))
        return EFLAGS
        
    else:
        return "NULL"


def getDisasmCapstone(addr):
    insn = None
    code = idc.get_bytes(addr, idc.get_item_size(addr))
    if not code:
        return None, None
    for i in md.disasm(code, addr):
        # addr = "0x%x" % i.address
        EFLAGS = getCapstoneInsFLAG(i)
        insn = "%s\t%s" % (i.mnemonic, i.op_str)
    return insn, EFLAGS



OPND_WRITE_FLAGS = {
    0: idaapi.CF_CHG1,
    1: idaapi.CF_CHG2,
    2: idaapi.CF_CHG3,
    3: idaapi.CF_CHG4,
    4: idaapi.CF_CHG5,
    5: idaapi.CF_CHG6,
}

OPND_READ_FLAGS = {
    0: idaapi.CF_USE1,
    1: idaapi.CF_USE2,
    2: idaapi.CF_USE3,
    3: idaapi.CF_USE4,
    4: idaapi.CF_USE5,
    5: idaapi.CF_USE6,
}

def parse_operands(ea, debug:bool) -> List[list]:
    insn = idautils.DecodeInstruction(ea)
    result = []

    feature = insn.get_canon_feature()

    for op in insn.ops:

        if op.type == idaapi.o_void:
            break

        if op.type == idaapi.o_reg: 
            op_id = idc.get_operand_value(ea,op.n) 
        else:
            op_id = 0


        is_write = int(feature & OPND_WRITE_FLAGS[op.n] > 0)
        is_read = int(feature & OPND_READ_FLAGS[op.n] > 0)

        result.append([op.type,op_id,is_read,is_write])
        
        if debug:
            action = '{}'.format('/'.join(filter(bool, ('read' if is_read else None, 'write' if is_write else None))))
            stringToPrint = f"Function <{idc.get_func_name(ea)}> Insn <{idc.GetDisasm(ea).split(';')[0]}> IType [{insn.itype}] Operand[{op.n}] Type [{op.type}] ID[{op_id}] <{idc.print_operand(ea, op.n)}> : {action}"
            print(stringToPrint)

    return insn.itype, result

def main(output_dir:str, debug:bool = True) -> None:
    os.makedirs(output_dir, exist_ok=True)

    func_info = [] 

    textStartEA = 0
    textEndEA = 0
    for seg in idautils.Segments():
        if (idc.get_segm_name(seg)==".text"):
            textStartEA = idc.get_segm_start(seg)
            textEndEA = idc.get_segm_end(seg)
            break
    
    for func in idautils.Functions(textStartEA, textEndEA):
        insn_and_opinfo = [] 

        flags = idc.get_func_attr(func, idc.FUNCATTR_FLAGS)
        if flags & idc.FUNC_LIB:
            if debug:
                print(hex(func), "FUNC_LIB", idc.get_func_name(func))
            continue
        start_time = time()
        for insn in idautils.FuncItems(func):
            disasm, eflags = getDisasmCapstone(insn)
            if disasm is None:
                continue
            itype, op_info = parse_operands(insn, debug)
            insn_and_opinfo.append([hex(insn),disasm,itype,op_info,eflags])
        end_time = time()
        print('Running for {} seconds.'.format(end_time-start_time))
        func_info.append(insn_and_opinfo)
    
    with open(os.path.join(output_dir, f'{ida_nalt.get_root_filename()}.pkl'), 'wb') as f:
        pickle.dump(func_info, f)    
    


if __name__ == '__main__':
    if len(idc.ARGV) < 2:
        print('\n\nGenerate Instructions with IDA Pro')
        print('\tIter through all .text functions and instructions to parse operands')
        print('\tExtract instruction type, operand read/write status, operand type and eflags.')
        print('\tUsage: /path/to/ida -A -Lida.log -S"{} output_dir" /path/to/binary\n\n'.format(idc.ARGV[0]))
        ida_pro.qexit(1)


    output_dir = idc.ARGV[1]
    os.makedirs(output_dir, exist_ok=True)

    ida_auto.auto_wait()
    
    main(output_dir, debug=False)
    
    ida_pro.qexit(0)