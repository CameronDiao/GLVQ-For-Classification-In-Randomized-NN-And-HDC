printf('lendo problema %s ...\n', problema);

n_entradas= 25; n_clases= 2; 
n_fich= 2; fich{1}= 'horse-colic.data'; n_patrons(1)= 300; fich{2}= 'horse-colic.test'; n_patrons(2)= 68;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;
i_entradas=[1 2 4:23 25 26 28];
n_col=28; t=cell(1,n_col);

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
   	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j=1:n_col
	  t{j}=fscanf(f,'%s',1);
	end
	for j=1:n_entradas
	  u = t{i_entradas(j)}; 
	  if u=='?'
		x(i_fich,i,j)= 0;
	  else
		x(i_fich,i,j)= str2double(u);
	  end
	end
	cl(i_fich,i)=str2double(t{24}) - 1;
  end
  fclose(f);
end
