printf('lendo problema %s ...\n', problema);

n_entradas= 32; n_clases= 2; n_fich= 1; fich{1}= 'trains-transformed.data'; n_patrons(1)= 10;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);
n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  t = fscanf(f,'%s',1);
	  if t ~= '-'
		if j==4 || j==9 || j==14 || j==19
		  val={'long','short'};
		elseif j==5 || j==10 || j==15 || j==20
		  val={'closedrect', 'dblopnrect', 'ellipse', 'engine', 'hexagon','jaggedtop', 'openrect', 'opentrap', 'slopetop', 'ushaped'};
		elseif j==7 || j==12 || j==17 || j==22
		  val={'circlelod', 'hexagonlod', 'rectanglod', 'trianglod'};
		else
		  x(i_fich,i,j) = str2double(t); continue
		end
		n=length(val);
		for k=1:n
		  if strcmp(t, val{k})
			x(i_fich,i,j)=k; break
		  end
		end
	  else
		x(i_fich,i,j) = 0;
	  end
	end
	t= fscanf(f,'%s',1);    	% lectura da clase
	if strcmp(t, 'east')
	  cl(i_fich,i) = 0;
	elseif strcmp(t, 'west')
	  cl(i_fich,i) = 1;
	else
	  error('clase %s descoñecida', t)
	end
  end
  fclose(f);
end
