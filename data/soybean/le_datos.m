printf('lendo problema %s ...\n', problema);

n_entradas= 35; n_clases= 18; 
n_fich= 2; fich{1}= 'soybean-large.data'; n_patrons(1)= 307; fich{2}= 'soybean-large.test'; n_patrons(2)= 376;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;

clase={'diaporthe-stem-canker', 'charcoal-rot', 'rhizoctonia-root-rot','phytophthora-rot', 'brown-stem-rot', 'powdery-mildew','downy-mildew','brown-spot', 'bacterial-blight','bacterial-pustule', 'purple-seed-stain', 'anthracnose','phyllosticta-leaf-spot', 'alternarialeaf-spot','frog-eye-leaf-spot', 'diaporthe-pod-&-stem-blight','cyst-nematode', 'herbicide-injury'};	
     
for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
   	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	t= fscanf(f,'%s',1);   % lectura de clase
	for j=1:n_clases
	  if strcmp(t, clase{j})
		cl(i_fich,i)=j-1; break
	  end
	end
%  	printf('cl= %i: ', cl(i_fich,i))
	for j = 1:n_entradas
	  t = fscanf(f,'%s',1);
	  if t ~= '?'
		x(i_fich,i,j) = str2double(t);
	  else
		x(i_fich,i,j) = 0;
	  end
%  	  printf('%g ', x(i_fich,i,j))
	end
%  	printf('\n')
%  	if i==2 exit end
  end
  fclose(f);
end
