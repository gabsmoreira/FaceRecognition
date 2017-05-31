import os
def create_file(nomes,n_class):
    os.mkdir("aulas")
    os.chdir("aulas")
    str_file_name = "aula{0}.txt".format(str(n_class))
    with open(str_file_name, 'w+') as fl:
        #writing names
        lines = 0
        for nomes in nomes:
            fl.write(nomes)
            fl.write(" ")
            fl.write(' \n')
            lines+=2
    return str_file_name,lines


def check_presence(filename,line_number, aluno):
    with open(filename, 'r') as fl:
        linhas = fl.readlines()
        for i in range(len(linhas)):
            if aluno in linhas[i]:
                if "check" not in linhas[i]:
                    str_linha = "{0} {1} check \n".format(linhas[i].split(" ")[0],linhas[i].split(" ")[1])
                linhas[i] = str_linha
    with open(filename,'w') as fl:
        fl.writelines(linhas)
