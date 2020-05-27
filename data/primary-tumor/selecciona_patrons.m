#!/usr/bin/octave -q

clear all
f1=fopen('primary-tumor-orixinal.data', 'r');
if -1==f1
  error('erro en fopen abrindo primary-tumor-orixinal.data');
end
f2=fopen('primary-tumor.data', 'w');
if -1==f1
  error('erro en fopen abrindo primary-tumor.data');
end
n_patrons=339; n_entradas=17; n_clases1=22; 
clase_suprimir=[6,9,10,15,16,20,21]; n_clases2= n_clases1 - sum(clase_suprimir);
for i=1:n_patrons
  t=fscanf(f1,'%i',1);
  if any(clase_suprimir==t)
	for j=1:n_entradas
	  fscanf(f1,'%s',1);
	end
	continue;
  end
  fprintf(f2,'%i ', t - sum(t > clase_suprimir));
  for j=1:n_entradas
	t=fscanf(f1,'%s',1);
	fprintf(f2, '%s ', t);
  end
  fprintf(f2,'\n');
end

fclose(f2);
fclose(f1);