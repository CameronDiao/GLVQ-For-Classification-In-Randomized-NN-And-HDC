printf('lendo problema %s ...\n', problema);

n_entradas= 6; n_clases= 18; n_fich= 1; fich{1}= 'krkopt.data'; n_patrons(1)= 28056;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;
val={'a','b','c','d','e','f','g','h'};
n=length(val); a=2/(n-1); b=(1+n)/(1-n);
clase={'draw','zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen'};
for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  if mod(j,2)==1
		t = fscanf(f,'%c',1);
		for k=1:n
		  if strcmp(t,val{k})
			x(i_fich,i,j)=a*k+b; break
		  end
		end
	  else
		x(i_fich,i,j) = fscanf(f, '%i',1);
	  end
	  fscanf(f,'%c',1);  % le e descarta a coma
	end	
	t = fscanf(f,'%s',1);
	for j=1:n_clases    	% lectura da clase
	  if strcmp(t,clase{j})
		cl(i_fich,i)=j-1; break
	  end
	end
	fscanf(f,'%c',1);
  end
  fclose(f);
end
